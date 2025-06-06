import os
import json
import copy
import math
from collections import OrderedDict
import torch
import torch.nn as nn
from numba import jit, prange
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import HGTConv, Linear, HeteroConv, SAGEConv
from utils.tools import (
    get_variance_level,
    get_phoneme_level_pitch,
    get_phoneme_level_energy,
    get_mask_from_lengths,
    get_mask_from_lengths_pad_left,
    pad_1D,
    pad,
)
from text.symbols import symbols
from .transformers.transformer import MultiHeadAttention, PositionwiseFeedForward
from .transformers.constants import PAD
from .transformers.blocks import get_sinusoid_encoding_table, Swish, LinearNorm, ConvNorm, ConvBlock, ConvBlock2D

from torch_geometric.data import Batch

import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 设置CUDA_DEBUGGER_SOFTWARE_PREEMPTION环境变量为1
# os.environ['CUDA_DEBUGGER_SOFTWARE_PREEMPTION'] = '1'
@jit(nopython=True)
def mas_width1(attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]): # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config, train_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        # self.duration_predictor = DurationPredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.binarization_start_steps = train_config["duration"]["binarization_start_steps"]
        if model_config["duration_modeling"]["learn_alignment"]:
            self.aligner = AlignmentEncoder(
                n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                n_att_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                n_text_channels=model_config["transformer"]["encoder_hidden"],
                temperature=model_config["duration_modeling"]["aligner_temperature"],
                multi_speaker=model_config["multi_speaker"],
                multi_emotion=model_config["multi_emotion"],
            )

        pitch_level_tag, energy_level_tag, self.pitch_feature_level, self.energy_feature_level = \
                                    get_variance_level(preprocess_config, model_config, data_loading=False)

        # Note that there is no pre-extracted phoneme-level variance features in unsupervised duration modeling.
        # Alternatively, we can use convolutional embedding instead of bucket-based embedding in such cases.
        self.use_conv_embedding = self.learn_alignment \
            and (self.pitch_feature_level == "phoneme_level" or self.energy_feature_level == "phoneme_level")
        if self.use_conv_embedding:
            kernel_size = model_config["variance_embedding"]["kernel_size"]
            self.pitch_embedding = ConvNorm(
                1, 
                model_config["transformer"]["encoder_hidden"],
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                bias=False,
                w_init_gain="tanh",
                transpose=True,
            )
            self.energy_embedding = ConvNorm(
                1,
                model_config["transformer"]["encoder_hidden"],
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                bias=False,
                w_init_gain="tanh",
                transpose=True,
            )
        else:
            pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
            energy_quantization = model_config["variance_embedding"]["energy_quantization"]
            n_bins = model_config["variance_embedding"]["n_bins"]
            assert pitch_quantization in ["linear", "log"]
            assert energy_quantization in ["linear", "log"]
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                pitch_min, pitch_max = stats[f"pitch_{pitch_level_tag}"][:2]
                energy_min, energy_max = stats[f"energy_{energy_level_tag}"][:2]

            if pitch_quantization == "log":
                self.pitch_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.pitch_bins = nn.Parameter(
                    torch.linspace(pitch_min, pitch_max, n_bins - 1),
                    requires_grad=False,
                )
            if energy_quantization == "log":
                self.energy_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.energy_bins = nn.Parameter(
                    torch.linspace(energy_min, energy_max, n_bins - 1),
                    requires_grad=False,
                )

            self.pitch_embedding = nn.Embedding(
                n_bins, model_config["transformer"]["encoder_hidden"]
            )
            self.energy_embedding = nn.Embedding(
                n_bins, model_config["transformer"]["encoder_hidden"]
            )

        # style_linear
        self.an_predict_style_linear = nn.Linear(768, 256)
        self.retri_style_weight = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.cur_text_semantics_weight = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.cur_audio_style_weight = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.an_predict_weight = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)



    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out).to(attn.device)

    def get_phoneme_level_pitch(self, duration, src_len, pitch_frame):
        return torch.from_numpy(
            pad_1D(
                [get_phoneme_level_pitch(dur[:len], var) for dur, len, var \
                        in zip(duration.int().cpu().numpy(), src_len.cpu().numpy(), pitch_frame.cpu().numpy())]
            )
        ).float().to(pitch_frame.device)

    def get_phoneme_level_energy(self, duration, src_len, energy_frame):
        return torch.from_numpy(
            pad_1D(
                [get_phoneme_level_energy(dur[:len], var) for dur, len, var \
                        in zip(duration.int().cpu().numpy(), src_len.cpu().numpy(), energy_frame.cpu().numpy())]
            )
        ).float().to(energy_frame.device)

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(target.unsqueeze(-1)) if self.use_conv_embedding \
                else self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(prediction.unsqueeze(-1)) if self.use_conv_embedding \
                else self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding =  self.energy_embedding(target.unsqueeze(-1)) if self.use_conv_embedding \
                else self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(prediction.unsqueeze(-1)) if self.use_conv_embedding \
                else self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        speaker_embedding,
        emotion_embedding,
        text,
        text_embedding,
        src_len,
        src_mask,
        mel,
        mel_len,
        mel_mask=None,
        max_len=None,
        style_cat_emb = None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        attn_prior=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
    ):
        x = text

        if style_cat_emb is not None:
            [
                retri_style_emb,
                cur_text_semantics_emb,
                cur_audio_style_emb,
                an_predict_emb
            ] = style_cat_emb

            an_predict_emb = self.an_predict_style_linear(an_predict_emb)
            style_final_emb = (retri_style_emb * self.retri_style_weight
                               + cur_text_semantics_emb * self.cur_text_semantics_weight
                               + cur_audio_style_emb * self.cur_audio_style_weight
                               + an_predict_emb * self.an_predict_weight)

            x = x + style_final_emb.unsqueeze(1).expand(
                -1, text.shape[1], -1
            )



        if speaker_embedding is not None:
            x = x + speaker_embedding.unsqueeze(1).expand(
                -1, text.shape[1], -1
            )
        if emotion_embedding is not None:
            x = x + emotion_embedding.unsqueeze(1).expand(
                -1, text.shape[1], -1
            )
        # x_dur = x.clone()

        log_duration_prediction = self.duration_predictor(x, src_mask)
        # log_duration_prediction = self.duration_predictor(x_dur, src_len, context_encoding, src_mask)
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            min=0,
        )

        # Trainig of unsupervised duration modeling
        attn_soft, attn_hard, attn_hard_dur, attn_logprob = None, None, None, None
        if attn_prior is not None:
            assert self.learn_alignment and duration_target is None and mel is not None
            attn_soft, attn_logprob = self.aligner(
                mel.transpose(1, 2),
                text_embedding.transpose(1, 2),
                src_mask.unsqueeze(-1),
                attn_prior.transpose(1, 2),
                speaker_embedding,
                emotion_embedding,
            )
            attn_hard = self.binarize_attention_parallel(attn_soft, src_len, mel_len)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        attn_out = (attn_soft, attn_hard, attn_hard_dur, attn_logprob)

        # Note that there is no pre-extracted phoneme-level variance features in unsupervised duration modeling.
        # Alternatively, we can use attn_hard_dur instead of duration_target for computing phoneme-level variances.
        output_1 = x.clone()
        if self.pitch_feature_level == "phoneme_level":
            if attn_prior is not None:
                pitch_target = self.get_phoneme_level_pitch(attn_hard_dur, src_len, pitch_target)
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
            output_1 = output_1 + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            if attn_prior is not None:
                energy_target = self.get_phoneme_level_energy(attn_hard_dur, src_len, energy_target)
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, e_control)
            output_1 = output_1 + energy_embedding
        x = output_1.clone()

        # Upsampling from src length to mel length
        if attn_prior is not None: # Trainig of unsupervised duration modeling
            if step < self.binarization_start_steps:
                A_soft = attn_soft.squeeze(1)
                x = torch.bmm(A_soft,x)
            else:
                x, mel_len = self.length_regulator(x, attn_hard_dur, max_len)
            duration_rounded = attn_hard_dur
        elif duration_target is not None: # Trainig of supervised duration modeling
            assert not self.learn_alignment and attn_prior is None
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else: # Inference
            assert attn_prior is None and duration_target is None
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        output_2 = x.clone()
        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)
            output_2 = output_2 + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mel_mask, e_control)
            output_2 = output_2 + energy_embedding
        x = output_2.clone()

        return (
            x,
            pitch_target,
            pitch_prediction,
            energy_target,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
            attn_out,
        )


class AlignmentEncoder(torch.nn.Module):
    """ Alignment Encoder for Unsupervised Duration Modeling """

    def __init__(self, 
                n_mel_channels,
                n_att_channels,
                n_text_channels,
                temperature,
                multi_speaker,
                multi_emotion):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu'
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_text_channels * 2,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                n_mel_channels,
                n_mel_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu',
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_mel_channels * 2,
                n_mel_channels,
                kernel_size=1,
                bias=True,
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_mel_channels,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        if multi_speaker:
            self.key_spk_proj = LinearNorm(n_text_channels, n_text_channels)
            self.query_spk_proj = LinearNorm(n_text_channels, n_mel_channels)
        if multi_emotion:
            self.key_emo_proj = LinearNorm(n_text_channels, n_text_channels)
            self.query_emo_proj = LinearNorm(n_text_channels, n_mel_channels)

    def forward(self, queries, keys, mask=None, attn_prior=None, speaker_embed=None, emotion_embed=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
            speaker_embed (torch.tensor): B x C tnesor of speaker embedding for multi-speaker scheme.
            emotion_embed (torch.tensor): B x C tnesor of emotion embedding for multi-emotion scheme.
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        if speaker_embed is not None:
            keys = keys + self.key_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, keys.shape[-1], -1
            )).transpose(1, 2)
            queries = queries + self.query_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        if emotion_embed is not None:
            keys = keys + self.key_emo_proj(emotion_embed.unsqueeze(1).expand(
                -1, keys.shape[-1], -1
            )).transpose(1, 2)
            queries = queries + self.query_emo_proj(emotion_embed.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            #print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)
            #print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config, positive_out=True):
        super(DurationPredictor, self).__init__()

        self.d_model = model_config["transformer"]["encoder_hidden"]
        self.d_hidden = model_config["variance_predictor"]["cond_dur_hidden"]

        self.max_seq_len = model_config["max_seq_len"]
        n_position = self.max_seq_len + 1
        n_head = model_config["variance_predictor"]["cond_dur_head"]
        d_w = self.d_hidden
        d_k = d_v = d_w // n_head
        d_inner = model_config["variance_predictor"]["conv_filter_size"]
        kernel_size = model_config["variance_predictor"]["conv_kernel_size"]
        dropout = model_config["variance_predictor"]["cond_dur_dropout"]

        self.cond_prj = LinearNorm(self.d_model, self.d_hidden)
        self.input_prj = nn.Sequential(
            ConvNorm(self.d_model, self.d_hidden, transpose=True),
            Swish(),
            LinearNorm(self.d_hidden, self.d_hidden),
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_hidden).unsqueeze(0),
            requires_grad=False,
        )
        self.layer_stack = nn.ModuleList(
            [
                LayerCondFFTBlock(
                    self.d_hidden, d_w, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(model_config["variance_predictor"]["cond_dur_layer"])
            ]
        )
        self.out = nn.Sequential(
            ConvNorm(self.d_hidden, 1, transpose=True),
            nn.ReLU() if positive_out else Swish(),
        )

    def forward(self, h_text, seq_len, h_context, mask):
        batch_size, max_len = h_text.shape[0], h_text.shape[1]

        # Input
        cond_g = self.cond_prj(h_context.unsqueeze(1)) # [B, 1, H]
        h_text = self.input_prj(h_text) # [B, seq_len, H]

        # Mask
        h_text = h_text.masked_fill(mask.unsqueeze(-1), 0)
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # Positional Encoding
        if not self.training and h_text.shape[1] > self.max_seq_len:
            output = h_text + get_sinusoid_encoding_table(
                h_text.shape[1], self.d_hidden
            )[: h_text.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                h_text.device
            )
        else:
            output = h_text + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        # Conditioned Duration Prediction
        for layer in self.layer_stack:
            output, _ = layer(
                output, cond_g, mask=mask, slf_attn_mask=slf_attn_mask
            )
        output = self.out(output).squeeze(-1)

        return output

class LayerCondFFTBlock(nn.Module):
    """ Layer Conditioning FFTBlock """

    def __init__(self, d_model, d_w, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(LayerCondFFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, layer_norm=False)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout, layer_norm=False
        )
        self.layer_norm_1 = StyleAdaptiveLayerNorm(d_w, d_model)
        self.layer_norm_2 = StyleAdaptiveLayerNorm(d_w, d_model)

    def forward(self, enc_input, cond_g, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.layer_norm_1(enc_output, cond_g)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = self.layer_norm_2(enc_output, cond_g)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class StyleAdaptiveLayerNorm(nn.Module):
    """ Style-Adaptive Layer Norm (SALN) """

    def __init__(self, w_size, hidden_size, bias=False):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            w_size,
            2 * hidden_size, # For both b (bias) g (gain) 
            bias,
        )

    def forward(self, h, cond_g):
        """
        h --- [B, T, H_m]
        cond_g --- [B, 1, H_w]
        o --- [B, T, H_m]
        """

        # Normalize Input Features
        mu, sigma = torch.mean(h, dim=-1, keepdim=True), torch.std(h, dim=-1, keepdim=True)
        y = (h - mu) / sigma # [B, T, H_m]

        # Get Bias and Gain
        b, g = torch.split(self.affine_layer(cond_g), self.hidden_size, dim=-1)  # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]

        # Perform Scailing and Shifting
        o = g * y + b # [B, T, H_m]

        return o


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        ConvNorm(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=(self.kernel - 1) // 2,
                            dilation=1,
                            transpose=True,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        ConvNorm(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=1,
                            dilation=1,
                            transpose=True,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class TextContextEncoder(nn.Module):
    """ Conversational Context Encoder """
    def __init__(self, preprocess_config, model_config):
        super(TextContextEncoder, self).__init__()
        d_model = model_config["transformer"]["encoder_hidden"] # 256
        d_cont_enc = model_config["history_encoder"]["context_hidden"] # 128
        num_layers = model_config["history_encoder"]["context_layer"] # 2
        dropout = model_config["history_encoder"]["context_dropout"] # 0.2
        text_dim = model_config["history_encoder"]["text_dim"] # 512

        self.linear_down = nn.Linear(text_dim, d_cont_enc)

        self.gru = nn.GRU(
            input_size=d_cont_enc,
            hidden_size=d_cont_enc,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.gru_linear = nn.Sequential(  # 256 -> 128
            nn.Linear(2*d_cont_enc, d_cont_enc),
            nn.ReLU()
        )

        self.context_linear = nn.Linear(d_cont_enc, d_model) # 128 -> 256
        self.context_attention = SLA(d_model)

    def forward(self, history_text_emb, history_lens):

        max_history_len = history_text_emb.shape[1]
        history_masks = get_mask_from_lengths_pad_left(history_lens, max_history_len)

        history_text_emb = self.linear_down(history_text_emb)

        # GRU
        context_enc = self.gru_linear(self.gru(history_text_emb)[0])
        context_enc = context_enc.masked_fill(history_masks.unsqueeze(-1), 0)

        # Encoding
        context_enc = self.context_attention(self.context_linear(context_enc), history_masks) # [B, d]

        return context_enc


class AudioContextEncoder(nn.Module):
    """ Conversational Context Encoder """
    def __init__(self, preprocess_config, model_config):
        super(AudioContextEncoder, self).__init__()
        d_model = model_config["transformer"]["encoder_hidden"]  # 256
        num_layers = model_config["history_encoder"]["context_layer"]  # 2
        dropout = model_config["history_encoder"]["context_dropout"]  # 0.2
        audio_dim = model_config["history_encoder"]["audio_dim"]  # 1280

        self.linear_down = nn.Linear(audio_dim, d_model)  # 1280 -> 256

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.gru_linear = nn.Sequential(  # 512 -> 256
            nn.Linear(2 * d_model, d_model),
            nn.ReLU()
        )

        self.context_linear = nn.Linear(d_model, d_model * 2)  # 256 ->512
        self.context_attention = SLA(d_model *2)

    def forward(self, history_text_emb, history_lens):

        max_history_len = history_text_emb.shape[1]
        history_masks = get_mask_from_lengths_pad_left(history_lens, max_history_len)

        history_text_emb = self.linear_down(history_text_emb)

        # GRU
        context_enc = self.gru_linear(self.gru(history_text_emb)[0])
        context_enc = context_enc.masked_fill(history_masks.unsqueeze(-1), 0)

        # Encoding
        context_enc = self.context_attention(self.context_linear(context_enc), history_masks)  # [B, d]

        return context_enc


class AnVectorPredictor(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(AnVectorPredictor, self).__init__()
        an_dim = model_config["history_encoder"]["an_dim"]
        self.text_context_encoder = TextContextEncoder(preprocess_config, model_config)
        self.audio_context_encoder = AudioContextEncoder(preprocess_config, model_config)

        self.cat_linear = nn.Sequential(  # 768 -> 768
            nn.Linear(an_dim, an_dim),
            nn.ReLU(),
            nn.Linear(an_dim, an_dim),
        )

    def forward(self, dia_text_emb, dia_audio_emb, dia_text_len, dia_audio_len):
        text_context = self.text_context_encoder(dia_text_emb, dia_text_len)
        audio_context = self.audio_context_encoder(dia_audio_emb, dia_audio_len)
        cat_emb = torch.cat((text_context, audio_context), dim=-1)
        an_predict_emb = self.cat_linear(cat_emb)
        return an_predict_emb

class SLA(nn.Module):
    """ Sequence Level Attention """
    def __init__(self, d_enc):
        super(SLA, self).__init__()
        self.linear = nn.Linear(d_enc, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoding, mask=None):

        attn = self.linear(encoding)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1), -np.inf)
            aux_mask = (attn == -np.inf).all(self.softmax.dim).unsqueeze(self.softmax.dim)
            attn = attn.masked_fill(aux_mask, 0) # Remove all -inf along softmax.dim
        score = self.softmax(attn).transpose(-2, -1) # [B, 1, T]
        fused_rep = torch.matmul(score, encoding).squeeze(1) # [B, d]

        return fused_rep



class SemanticsMultiGranularityHeteroGraph(nn.Module):
    def __init__(self, hidden_channels):
        super(SemanticsMultiGranularityHeteroGraph, self).__init__()
        self.linear_conversation = Linear(512, 256)
        self.linear_sentence = Linear(512, 256)
        self.linear_word = Linear(768, 256)
        self.heteroConv1 = HeteroConv({
            ("coversation", "connect", "sentence"): SAGEConv(hidden_channels, hidden_channels),
            ("sentence", "connect", "sentence"): SAGEConv(hidden_channels, hidden_channels),
            ("sentence", "connect", "word"): SAGEConv(hidden_channels, hidden_channels),
            ("word", "connect", "word"): SAGEConv(hidden_channels, hidden_channels),
            ("sentence", "rev_connect", "coversation"): SAGEConv(hidden_channels, hidden_channels),
            ("word", "rev_connect", "sentence"): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')


    def forward(self, x_dict=None, edge_index_dict=None):
        x_dict["coversation"] = self.linear_conversation(x_dict["coversation"])
        x_dict["sentence"] = self.linear_sentence(x_dict["sentence"])
        x_dict["word"] = self.linear_word(x_dict["word"])

        x_dict = self.heteroConv1(x_dict, edge_index_dict)
        return x_dict


class TextMultiGranularityHeteroGraphFusion(nn.Module):
    def __init__(self):
        super(TextMultiGranularityHeteroGraphFusion, self).__init__()
        self.sentence_fusion = NodeFeatureFusion(d_model=256, hidden_size=256)
        self.word_fusion = NodeFeatureFusion(d_model=256, hidden_size=256)
        self.cat_linear = nn.Sequential(  # 768 -> 768
            nn.Linear(256 * 3, 256 * 2),
            nn.ReLU(),
            nn.Linear(256 * 2, 256),
        )

    def forward(self, text_cove_emb, text_sen_emb, text_word_emb):
        flag = False
        if text_cove_emb.dim() == 3:
            flag = True
        if(flag):
            batch = text_cove_emb.shape[0]
            K = text_cove_emb.shape[1]
            text_cove_emb = text_cove_emb.view(batch * K, text_cove_emb.shape[2])
            text_sen_emb = text_sen_emb.view(batch * K, text_sen_emb.shape[2], text_sen_emb.shape[3])
            text_word_emb = text_word_emb.view(batch * K, text_word_emb.shape[2], text_word_emb.shape[3])


        sentences_feat = self.sentence_fusion(text_sen_emb)
        words_feat = self.word_fusion(text_word_emb)
        output = torch.cat([text_cove_emb,sentences_feat,words_feat], dim=-1)
        output = self.cat_linear(output)

        if(flag):
            output = output.view(batch, K, -1)

        return output

class StyleMultiGranularityHeteroGraph(nn.Module):
    def __init__(self, hidden_channels):
        super(StyleMultiGranularityHeteroGraph, self).__init__()
        self.linear_conversation = Linear(1280, 256)
        self.linear_sentence = Linear(1280, 256)
        self.linear_word = Linear(768, 256)
        self.heteroConv1 = HeteroConv({
            ("coversation", "connect", "sentence"): SAGEConv(hidden_channels, hidden_channels),
            ("sentence", "connect", "sentence"): SAGEConv(hidden_channels, hidden_channels),
            ("sentence", "connect", "word"): SAGEConv(hidden_channels, hidden_channels),
            ("word", "connect", "word"): SAGEConv(hidden_channels, hidden_channels),
            ("sentence", "rev_connect", "coversation"): SAGEConv(hidden_channels, hidden_channels),
            ("word", "rev_connect", "sentence"): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')


    def forward(self, x_dict=None, edge_index_dict=None):
        x_dict["coversation"] = self.linear_conversation(x_dict["coversation"])
        x_dict["sentence"] = self.linear_sentence(x_dict["sentence"])
        x_dict["word"] = self.linear_word(x_dict["word"])

        x_dict = self.heteroConv1(x_dict, edge_index_dict)
        return x_dict


class StyleMultiGranularityHeteroGraphFusion(nn.Module):
    def __init__(self):
        super(StyleMultiGranularityHeteroGraphFusion, self).__init__()
        self.sentence_fusion = NodeFeatureFusion(d_model=256, hidden_size=256)
        self.word_fusion = NodeFeatureFusion(d_model=256, hidden_size=256)
        self.cat_linear = nn.Sequential(  # 768 -> 768
            nn.Linear(256 * 3, 256 * 2),
            nn.ReLU(),
            nn.Linear(256 * 2, 256),
        )

    def forward(self, audio_cove_emb, audio_sen_emb, audio_word_emb):
        flag = False
        if audio_cove_emb.dim() == 3:
            flag = True
        if(flag):
            batch = audio_cove_emb.shape[0]
            K = audio_cove_emb.shape[1]
            audio_cove_emb = audio_cove_emb.view(batch * K, audio_cove_emb.shape[2])
            audio_sen_emb = audio_sen_emb.view(batch * K, audio_sen_emb.shape[2], audio_sen_emb.shape[3])
            audio_word_emb = audio_word_emb.view(batch * K, audio_word_emb.shape[2], audio_word_emb.shape[3])


        sentences_feat = self.sentence_fusion(audio_sen_emb)
        words_feat = self.word_fusion(audio_word_emb)
        output = torch.cat([audio_cove_emb,sentences_feat,words_feat], dim=-1)
        output = self.cat_linear(output)

        if(flag):
            output = output.view(batch, K, -1)
        return output
class NodeFeatureFusion(nn.Module):
    def __init__(self, d_model, hidden_size):
        super(NodeFeatureFusion, self).__init__()
        self.d_model = d_model
        self.bidirectional = True
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(self.d_model, hidden_size, 1, batch_first=True,
                                  bidirectional=self.bidirectional)
        self.cat_linear = nn.Sequential(  # 768 -> 768
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 256),
        )


    def forward(self, input):
        output, (hn, cn) = self.lstm(input)
        hn = torch.cat([hn[0], hn[1]], dim=1)
        hn = self.cat_linear(hn)
        return hn
