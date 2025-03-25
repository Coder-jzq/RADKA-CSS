import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import PostNet, VarianceAdaptor, SemanticsMultiGranularityHeteroGraph, TextMultiGranularityHeteroGraphFusion, StyleMultiGranularityHeteroGraph, StyleMultiGranularityHeteroGraphFusion, AnVectorPredictor
from utils.tools import get_mask_from_lengths

from torch_geometric.data import Batch
from torch_geometric.nn import HypergraphConv


device = torch.device('cuda:{:d}'.format(5))

class CompTransTTS(nn.Module):
    """ CompTransTTS """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTS, self).__init__()
        self.model_config = model_config
        if model_config["block_type"] == "transformer":
            from .transformers.transformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "lstransformer":
        #     from .transformers.lstransformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "fastformer":
        #     from .transformers.fastformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "conformer":
        #     from .transformers.conformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "reformer":
        #     from .transformers.reformer import TextEncoder, Decoder
        else:
            raise ValueError("Unsupported Block Type: {}".format(model_config["block_type"]))

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = self.emotion_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )
        if model_config["multi_emotion"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                n_emotion = len(json.load(f))
            self.emotion_emb = nn.Embedding(
                n_emotion,
                model_config["transformer"]["encoder_hidden"],
            )
        self.history_type = model_config["history_encoder"]["type"]

        self.an_predictor =  AnVectorPredictor(preprocess_config, model_config)



        self.style_flag=model_config['style_flag']
        self.linguistic_flag = model_config['style_flag']

        self.semantics_multi_granularity_hetero_encoder = SemanticsMultiGranularityHeteroGraph(hidden_channels=256)
        self.text_multi_granularity_hetero_fusion = TextMultiGranularityHeteroGraphFusion()

        self.style_multi_granularity_hetero_encoder = StyleMultiGranularityHeteroGraph(hidden_channels=256)
        self.audio_multi_granularity_hetero_fusion = StyleMultiGranularityHeteroGraphFusion()


        self.K = train_config["p_n_sample"]["K"]

    def weighted_sum(self, t, T, S):
        # (16, 256) -> (16, 1, 256)
        t = t.unsqueeze(1)

        # t (16, 1, 256) -> (16, 256, 1)
        t = t.transpose(-2, -1)

        # (16, K, 256) @ (16, 256, 1) -> (16, K, 1)
        hidd = T @ t

        # hidd -> (16, K, 1)
        softmax = torch.nn.Softmax(dim=1)
        W = softmax(hidd)

        # (16, K, 1) -> (16, 1, K)
        W = W.transpose(-2, -1)

        # (16, 1, K) @ (16, K, 256) -> (16, 1, 256)
        final_style_embedding = W @ S

        # (16, 1, 256) -> (16, 256)
        final_style_embedding = final_style_embedding.squeeze(1)

        return final_style_embedding
    def pad_left_2d_max_len(self, pad_list):
        max_len = max(t.shape[0] for t in pad_list)

        lengths = [t.shape[0] for t in pad_list]

        padded_list = [torch.nn.functional.pad(t, (0, 0, max_len - t.shape[0], 0), mode='constant', value=0)
                       if t.shape[0] < max_len else t for t in pad_list]
        return padded_list, lengths

    def pad_left_3d_max_len(self, pad_list):
        max_len = max(t.shape[0] for sublist in pad_list for t in sublist)

        lengths = [[t.shape[0] for t in sublist] for sublist in pad_list]

        padded_list = [
            [torch.nn.functional.pad(t, (0, 0, max_len - t.shape[0], 0), mode='constant', value=0)
             if t.shape[0] < max_len else t for t in sublist]
            for sublist in pad_list
        ]

        return padded_list, lengths

    def forward(
        self,
        xxx1,
        xxx2,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        attn_priors=None,
        spker_embeds=None,
        emotions=None,
        cur_sen_dig_texts=None,
        cur_sen_dig_audios=None,
        aN_ground_truth=None,
        cur_sen_dig_text_len=None,
        cur_sen_dig_audio_len=None,
        text_positive_graphs=None,
        audio_positive_graphs=None,
        text_negative_graphs=None,
        audio_negative_graphs=None,
        cur_text_graphs=None,
        cur_audio_graphs=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        texts, text_embeds = self.encoder(texts, src_masks)


        an_predict_emb =  self.an_predictor(cur_sen_dig_texts, cur_sen_dig_audios, cur_sen_dig_text_len, cur_sen_dig_audio_len)



        cur_text_graph_batch = Batch.from_data_list(cur_text_graphs)
        cur_text_graph_batch_enc = self.semantics_multi_granularity_hetero_encoder(cur_text_graph_batch.x_dict, cur_text_graph_batch.edge_index_dict)
        cur_text_graph_batch.x_dict = cur_text_graph_batch_enc
        cur_text_graph_batch["coversation"].x = cur_text_graph_batch_enc["coversation"]
        cur_text_graph_batch["sentence"].x = cur_text_graph_batch_enc["sentence"]
        cur_text_graph_batch["word"].x = cur_text_graph_batch_enc["word"]

        cur_text_graph_list = cur_text_graph_batch.to_data_list()
        cur_text_coversation_list = []
        cur_text_sentence_list = []
        cur_text_word_list = []
        for cur_text_graph in cur_text_graph_list:
            cur_text_coversation_list.append(cur_text_graph["coversation"].x)
            cur_text_sentence_list.append(cur_text_graph["sentence"].x)
            cur_text_word_list.append(cur_text_graph["word"].x)

        cur_text_sentence_list_pad, cur_text_sen_pad_lens = self.pad_left_2d_max_len(cur_text_sentence_list)
        cur_text_word_list_pad, cur_text_word_pad_lens = self.pad_left_2d_max_len(cur_text_word_list)

        cur_text_coversation_embs = torch.cat(cur_text_coversation_list)
        cur_text_sentence_pad_embs = torch.stack(cur_text_sentence_list_pad)
        cur_text_word_pad_embs = torch.stack(cur_text_word_list_pad)

        cur_text_semantics_emb = self.text_multi_granularity_hetero_fusion(cur_text_coversation_embs, cur_text_sentence_pad_embs, cur_text_word_pad_embs)



        cur_audio_graph_batch = Batch.from_data_list(cur_audio_graphs)
        cur_audio_graph_batch_enc = self.style_multi_granularity_hetero_encoder(cur_audio_graph_batch.x_dict, cur_audio_graph_batch.edge_index_dict)
        cur_audio_graph_batch.x_dict = cur_audio_graph_batch_enc
        cur_audio_graph_batch["coversation"].x = cur_audio_graph_batch_enc["coversation"]
        cur_audio_graph_batch["sentence"].x = cur_audio_graph_batch_enc["sentence"]
        cur_audio_graph_batch["word"].x = cur_audio_graph_batch_enc["word"]
        cur_audio_graph_list = cur_audio_graph_batch.to_data_list()
        cur_audio_coversation_list = []
        cur_audio_sentence_list = []
        cur_audio_word_list = []
        for cur_audio_graph in cur_audio_graph_list:
            cur_audio_coversation_list.append(cur_audio_graph["coversation"].x)
            cur_audio_sentence_list.append(cur_audio_graph["sentence"].x)
            cur_audio_word_list.append(cur_audio_graph["word"].x)
        cur_audio_sentence_list_pad, cur_audio_sen_pad_lens = self.pad_left_2d_max_len(cur_audio_sentence_list)
        cur_audio_word_list_pad, cur_audio_word_pad_lens = self.pad_left_2d_max_len(cur_audio_word_list)

        cur_audio_coversation_embs = torch.cat(cur_audio_coversation_list)
        cur_audio_sentence_pad_embs = torch.stack(cur_audio_sentence_list_pad)
        cur_audio_word_pad_embs = torch.stack(cur_audio_word_list_pad)

        cur_audio_style_emb = self.audio_multi_granularity_hetero_fusion(cur_audio_coversation_embs, cur_audio_sentence_pad_embs, cur_audio_word_pad_embs)





        posi_text_graphs = [text_graph for text_graphs in text_positive_graphs for text_graph in text_graphs]
        nega_text_graphs = [text_graph for text_graphs in text_negative_graphs for text_graph in text_graphs]

        posi_nega_text_graphs = posi_text_graphs + nega_text_graphs
        posi_nega_text_graph_batch = Batch.from_data_list(posi_nega_text_graphs)

        posi_nega_text_graph_batch_enc = self.semantics_multi_granularity_hetero_encoder(posi_nega_text_graph_batch.x_dict, posi_nega_text_graph_batch.edge_index_dict)
        posi_nega_text_graph_batch.x_dict = posi_nega_text_graph_batch_enc
        posi_nega_text_graph_batch["coversation"].x = posi_nega_text_graph_batch_enc["coversation"]
        posi_nega_text_graph_batch["sentence"].x = posi_nega_text_graph_batch_enc["sentence"]
        posi_nega_text_graph_batch["word"].x = posi_nega_text_graph_batch_enc["word"]

        posi_nega_text_graph_list = posi_nega_text_graph_batch.to_data_list()
        p_n_g_mid = len(posi_nega_text_graph_list)//2
        posi_text_graph_list = posi_nega_text_graph_list[:p_n_g_mid]
        nega_text_graph_list = posi_nega_text_graph_list[p_n_g_mid:]

        posi_text_coversation_lists = []
        posi_text_sentence_lists = []
        posi_text_word_lists = []
        k_posi_text_iter = lambda posi_text_graph_list, k: (posi_text_graph_list[i:i + k] for i in range(0, len(posi_text_graph_list), k))
        for k_posi_text_graphs in k_posi_text_iter(posi_text_graph_list, self.K):
            posi_text_coversation_item_list = []
            posi_text_sentence_item_list = []
            posi_text_word_item_list = []
            for posi_text_graph in k_posi_text_graphs:
                posi_text_coversation_item_list.append(posi_text_graph["coversation"].x)
                posi_text_sentence_item_list.append(posi_text_graph["sentence"].x)
                posi_text_word_item_list.append(posi_text_graph["word"].x)
            posi_text_coversation_lists.append(posi_text_coversation_item_list)
            posi_text_sentence_lists.append(posi_text_sentence_item_list)
            posi_text_word_lists.append(posi_text_word_item_list)

        nega_text_coversation_lists = []
        nega_text_sentence_lists = []
        nega_text_word_lists = []
        k_nega_text_iter = lambda nega_text_graph_list, k: (nega_text_graph_list[i:i + k] for i in range(0, len(nega_text_graph_list), k))
        for k_nega_text_graphs in k_nega_text_iter(nega_text_graph_list, self.K):
            nega_text_coversation_item_list = []
            nega_text_sentence_item_list = []
            nega_text_word_item_list = []
            for nega_text_graph in k_nega_text_graphs:
                nega_text_coversation_item_list.append(nega_text_graph["coversation"].x)
                nega_text_sentence_item_list.append(nega_text_graph["sentence"].x)
                nega_text_word_item_list.append(nega_text_graph["word"].x)
            nega_text_coversation_lists.append(nega_text_coversation_item_list)
            nega_text_sentence_lists.append(nega_text_sentence_item_list)
            nega_text_word_lists.append(nega_text_word_item_list)

        posi_text_sentence_pad_lists, posi_text_sen_lens = self.pad_left_3d_max_len(posi_text_sentence_lists)
        posi_text_word_pad_lists, posi_text_word_lens = self.pad_left_3d_max_len(posi_text_word_lists)
        nega_text_sentence_pad_lists, nega_text_sen_lens = self.pad_left_3d_max_len(nega_text_sentence_lists)
        nega_text_word_pad_lists, nega_text_word_lens = self.pad_left_3d_max_len(nega_text_word_lists)

        posi_text_coversation_batch_k_emb = torch.stack([torch.cat(sublist) for sublist in posi_text_coversation_lists])
        posi_text_sentence_batch_k_emb = torch.stack([torch.stack(sublist) for sublist in posi_text_sentence_pad_lists])
        posi_text_word_batch_k_emb = torch.stack([torch.stack(sublist) for sublist in posi_text_word_pad_lists])
        nega_text_coversation_batch_k_emb = torch.stack([torch.cat(sublist) for sublist in nega_text_coversation_lists])
        nega_text_sentence_batch_k_emb = torch.stack([torch.stack(sublist) for sublist in nega_text_sentence_pad_lists])
        nega_text_word_batch_k_emb = torch.stack([torch.stack(sublist) for sublist in nega_text_word_pad_lists])

        posi_text_semantics_emb = self.text_multi_granularity_hetero_fusion(posi_text_coversation_batch_k_emb, posi_text_sentence_batch_k_emb, posi_text_word_batch_k_emb)
        nega_text_semantics_emb = self.text_multi_granularity_hetero_fusion(nega_text_coversation_batch_k_emb, nega_text_sentence_batch_k_emb, nega_text_word_batch_k_emb)



        posi_audio_graphs = [audio_graph for audio_graphs in audio_positive_graphs for audio_graph in audio_graphs]
        nega_audio_graphs = [audio_graph for audio_graphs in audio_negative_graphs for audio_graph in audio_graphs]
        posi_nega_audio_graphs = posi_audio_graphs + nega_audio_graphs
        posi_nega_audio_graph_batch = Batch.from_data_list(posi_nega_audio_graphs)
        posi_nega_audio_graph_batch_enc = self.style_multi_granularity_hetero_encoder(posi_nega_audio_graph_batch.x_dict, posi_nega_audio_graph_batch.edge_index_dict)
        posi_nega_audio_graph_batch.x_dict = posi_nega_audio_graph_batch_enc
        posi_nega_audio_graph_batch["coversation"].x = posi_nega_audio_graph_batch_enc["coversation"]
        posi_nega_audio_graph_batch["sentence"].x = posi_nega_audio_graph_batch_enc["sentence"]
        posi_nega_audio_graph_batch["word"].x = posi_nega_audio_graph_batch_enc["word"]
        posi_nega_audio_graph_list = posi_nega_audio_graph_batch.to_data_list()
        p_n_g_mid = len(posi_nega_audio_graph_list)//2
        posi_audio_graph_list = posi_nega_audio_graph_list[:p_n_g_mid]
        nega_audio_graph_list = posi_nega_audio_graph_list[p_n_g_mid:]
        posi_audio_coversation_lists = []
        posi_audio_sentence_lists = []
        posi_audio_word_lists = []
        k_posi_audio_iter = lambda posi_audio_graph_list, k: (posi_audio_graph_list[i:i + k] for i in range(0, len(posi_audio_graph_list), k))
        for k_posi_audio_graphs in k_posi_audio_iter(posi_audio_graph_list, self.K):
            posi_audio_coversation_item_list = []
            posi_audio_sentence_item_list = []
            posi_audio_word_item_list = []
            for posi_audio_graph in k_posi_audio_graphs:
                posi_audio_coversation_item_list.append(posi_audio_graph["coversation"].x)
                posi_audio_sentence_item_list.append(posi_audio_graph["sentence"].x)
                posi_audio_word_item_list.append(posi_audio_graph["word"].x)
            posi_audio_coversation_lists.append(posi_audio_coversation_item_list)
            posi_audio_sentence_lists.append(posi_audio_sentence_item_list)
            posi_audio_word_lists.append(posi_audio_word_item_list)
        nega_audio_coversation_lists = []
        nega_audio_sentence_lists = []
        nega_audio_word_lists = []
        k_nega_audio_iter = lambda nega_audio_graph_list, k: (nega_audio_graph_list[i:i + k] for i in range(0, len(nega_audio_graph_list), k))
        for k_nega_audio_graphs in k_nega_audio_iter(nega_audio_graph_list, self.K):
            nega_audio_coversation_item_list = []
            nega_audio_sentence_item_list = []
            nega_audio_word_item_list = []
            for nega_audio_graph in k_nega_audio_graphs:
                nega_audio_coversation_item_list.append(nega_audio_graph["coversation"].x)
                nega_audio_sentence_item_list.append(nega_audio_graph["sentence"].x)
                nega_audio_word_item_list.append(nega_audio_graph["word"].x)
            nega_audio_coversation_lists.append(nega_audio_coversation_item_list)
            nega_audio_sentence_lists.append(nega_audio_sentence_item_list)
            nega_audio_word_lists.append(nega_audio_word_item_list)
        posi_audio_sentence_pad_lists, posi_audio_sen_lens = self.pad_left_3d_max_len(posi_audio_sentence_lists)
        posi_audio_word_pad_lists, posi_audio_word_lens = self.pad_left_3d_max_len(posi_audio_word_lists)
        nega_audio_sentence_pad_lists, nega_audio_sen_lens = self.pad_left_3d_max_len(nega_audio_sentence_lists)
        nega_audio_word_pad_lists, nega_audio_word_lens = self.pad_left_3d_max_len(nega_audio_word_lists)
        posi_audio_coversation_batch_k_emb = torch.stack([torch.cat(sublist) for sublist in posi_audio_coversation_lists])
        posi_audio_sentence_batch_k_emb = torch.stack([torch.stack(sublist) for sublist in posi_audio_sentence_pad_lists])
        posi_audio_word_batch_k_emb = torch.stack([torch.stack(sublist) for sublist in posi_audio_word_pad_lists])
        nega_audio_coversation_batch_k_emb = torch.stack([torch.cat(sublist) for sublist in nega_audio_coversation_lists])
        nega_audio_sentence_batch_k_emb = torch.stack([torch.stack(sublist) for sublist in nega_audio_sentence_pad_lists])
        nega_audio_word_batch_k_emb = torch.stack([torch.stack(sublist) for sublist in nega_audio_word_pad_lists])

        posi_audio_style_emb = self.audio_multi_granularity_hetero_fusion(posi_audio_coversation_batch_k_emb, posi_audio_sentence_batch_k_emb, posi_audio_word_batch_k_emb)
        nega_audio_style_emb = self.audio_multi_granularity_hetero_fusion(nega_audio_coversation_batch_k_emb, nega_audio_sentence_batch_k_emb, nega_audio_word_batch_k_emb)



        retri_style_emb = self.weighted_sum(
            cur_text_semantics_emb,
            posi_text_semantics_emb,
            posi_audio_style_emb
        )
        style_cat_emb = [
            retri_style_emb,
            cur_text_semantics_emb,
            cur_audio_style_emb,
            an_predict_emb
        ]

        speaker_embeds = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_embeds = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_embeds = self.speaker_emb(spker_embeds) # [B, H]

        emotion_embeds = None
        if self.emotion_emb is not None:
            emotion_embeds = self.emotion_emb(emotions)

        (
            output,
            p_targets,
            p_predictions,
            e_targets,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            attn_outs,
        ) = self.variance_adaptor(
            speaker_embeds,
            emotion_embeds,
            texts,
            text_embeds,
            src_lens,
            src_masks,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            style_cat_emb,
            p_targets,
            e_targets,
            d_targets,
            attn_priors,
            p_control,
            e_control,
            d_control,
            step,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output


        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            p_targets,
            e_targets,
        ), (
            aN_ground_truth,  #  (16, 768)
            an_predict_emb,   #    (16, 768)
            posi_text_semantics_emb,   #    (16, 5, 256)
            nega_text_semantics_emb,   #    (16, 5, 256)
            posi_audio_style_emb,   #    (16, 5, 256)
            nega_audio_style_emb   #    (16, 5, 256)
        )
