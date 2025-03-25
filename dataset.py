import glob
import json
import math
import os
import scipy as sp
import scipy.sparse as ssp
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

from text import text_to_sequence
from utils.tools import get_variance_level, pad_1D, pad_2D, pad_3D

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from text import cleaners

import time

rank = "5"


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.sub_dir_name = preprocess_config["path"]["sub_dir_name"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        self.load_emotion = model_config["multi_emotion"]
        self.history_type = model_config["history_encoder"]["type"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.pitch_level_tag, self.energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)

        self.basename, self.speaker, self.text, self.raw_text, self.emotion = self.process_meta(
            filename
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        self.topK = train_config["p_n_sample"]["K"]

        self.retrieval_index_dir =  self.preprocessed_path + "/hetero_graph/retrieval_index/"
        self.retrieval_graph_dir =  self.preprocessed_path + "/hetero_graph/graph_text_audio_spkcat/"


    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[self.emotion[idx]] if self.load_emotion else None
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel_{}".format(self.pitch_level_tag),
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch_{}".format(self.pitch_level_tag),
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy_{}".format(self.energy_level_tag),
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        if self.learn_alignment:
            attn_prior_path = os.path.join(
                self.preprocessed_path,
                "attn_prior",
                "{}-attn_prior-{}.npy".format(speaker, basename),
            )
            attn_prior = np.load(attn_prior_path)
            duration = None
        else:
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)
            attn_prior = None
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None



        dialog_index = basename.split('_')[-1]
        turn_index = basename.split('_')[0]
        retrieval_dialog_path = self.retrieval_index_dir + dialog_index + "turn" + turn_index + ".json"
        with open(retrieval_dialog_path, 'r', encoding='utf-8') as file:
            retri_filename = json.load(file)

        topK_filename = retri_filename['topK_file_name'][:self.topK]
        bottomK_filename = retri_filename['bottomK_file_name'][:self.topK]


        text_positive_graph = []
        audio_positive_graph = []

        for d_path in topK_filename:
            data_path = self.retrieval_graph_dir + d_path[:-4] + ".pkl"
            with open(data_path, 'rb') as file:
                data = pickle.load(file)
            text_positive_graph.append(data['graph_text'])
            audio_positive_graph.append(data['graph_audio'])

        text_negative_graph = []
        audio_negative_graph = []

        for d_path in bottomK_filename:
            data_path = self.retrieval_graph_dir + d_path[:-4] + ".pkl"
            with open(data_path, 'rb') as file:
                data = pickle.load(file)
            text_negative_graph.append(data['graph_text'])
            audio_negative_graph.append(data['graph_audio'])

        dialog = basename.split("_")[2].strip("d")
        turn = int(basename.split("_")[0])
        cur_spk_id = basename.split("_")[1]
        history_len = min(self.max_history_len, turn)
        history_text = list()
        history_text_emb = list()
        history_text_len = list()
        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        history_mel_len = list()
        history = None

        if self.history_type != "none":
            history_basenames = sorted([tg_path.replace(".wav", "") for tg_path in os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
            history_basenames = history_basenames[:turn][-history_len:]

            graph_text = None
            graph_audio = None
            if turn != 0:
                text_path_g = self.preprocessed_path + "/text_emb/"
                text_words_tod_path_g = self.preprocessed_path + "/text_words_tod_emb/"
                waviemocap_path_g = self.preprocessed_path + "/audio_sencat_emb/"
                audio_word_path_g = self.preprocessed_path + "/audio_word_emb/"

                waviemocap_cur_path_g = self.preprocessed_path + "/waviemocap_emb/"


                text_summary_path_g = self.preprocessed_path + "/text_summary_emb_iter/d" + dialog + "turn" + str(
                    turn) + ".npy"
                text_summary_emb_g = np.load(text_summary_path_g)
                text_emb_g_list = []
                text_words_tod_emb_g_list = []
                audio_emb_g_list = []
                audio_words_emb_g_list = []

                for f_name in history_basenames:

                    f_list = f_name.split("_")
                    text_path_g_re = text_path_g + f_list[1] + "-text_emb-" + f_list[0] + "_" + f_list[
                        1] + "_d" + dialog + ".npy"
                    text_emb_g = np.load(text_path_g_re)
                    text_emb_g_list.append(text_emb_g)
                    text_words_tod_path_g_re = text_words_tod_path_g + f_list[1] + "-text_words_tod_emb-" + f_list[
                        0] + "_" + f_list[1] + "_d" + dialog + ".npy"
                    text_words_tod_emb_g = np.load(text_words_tod_path_g_re)
                    text_words_tod_emb_g_list.append(text_words_tod_emb_g)
                    waviemocap_path_g_re = waviemocap_path_g + f_list[1] + "-audio_sen_emb-" + f_list[0] + "_" + \
                                           f_list[1] + "_d" + dialog + ".npy"
                    audio_emb_g = np.load(waviemocap_path_g_re)
                    audio_emb_g_list.append(audio_emb_g)
                    audio_word_path_g_re = audio_word_path_g + f_list[1] + "-audio_word_emb-" + f_list[0] + "_" + \
                                           f_list[1] + "_d" + dialog + ".npy"
                    audio_word_emb_g = np.load(audio_word_path_g_re)
                    audio_words_emb_g_list.append(audio_word_emb_g)

                f_list_cur = basename.split("_")
                text_path_g_re = text_path_g + f_list_cur[1] + "-text_emb-" + f_list_cur[0] + "_" + f_list_cur[
                    1] + "_d" + dialog + ".npy"
                text_emb_g = np.load(text_path_g_re)
                text_emb_g_list.append(text_emb_g)

                text_words_tod_path_g_re = text_words_tod_path_g + f_list_cur[1] + "-text_words_tod_emb-" + f_list_cur[
                    0] + "_" + f_list_cur[1] + "_d" + dialog + ".npy"
                text_words_tod_emb_g = np.load(text_words_tod_path_g_re)
                text_words_tod_emb_g_list.append(text_words_tod_emb_g)

                text_summary_emb_g = torch.tensor(text_summary_emb_g)
                text_summary_emb_g = torch.unsqueeze(text_summary_emb_g, dim=0)

                text_emb_tensor_g = np.stack(text_emb_g_list)
                text_emb_tensor_g = torch.tensor(text_emb_tensor_g)

                text_words_emb_tensor_g = np.concatenate(text_words_tod_emb_g_list, axis=0)
                text_words_emb_tensor_g = torch.tensor(text_words_emb_tensor_g)

                audio_emb_tensor_g = np.stack(audio_emb_g_list)
                audio_conversational_emb_g = np.mean(audio_emb_tensor_g, axis=0)
                audio_emb_tensor_g = torch.tensor(audio_emb_tensor_g)
                audio_conversational_emb_g = torch.tensor(audio_conversational_emb_g)
                audio_conversational_emb_g = torch.unsqueeze(audio_conversational_emb_g, dim=0)

                audio_words_emb_tensor_g = np.concatenate(audio_words_emb_g_list, axis=0)
                audio_words_emb_tensor_g = torch.tensor(audio_words_emb_tensor_g)
                audio_words_emb_tensor_g = torch.squeeze(audio_words_emb_tensor_g, dim=1)

                cov_to_sen = []
                for idx in range(len(text_emb_g_list)):
                    cov_to_sen.append([0, idx])
                sen_to_sen = []
                for idx in range(len(text_emb_g_list) - 1):
                    sen_to_sen.append([idx, idx + 1])
                if (len(sen_to_sen) == 0):
                    sen_to_sen.append([0, 0])
                sen_to_word = []
                count_w = 0
                word_to_word = []
                for idx1 in range(len(text_emb_g_list)):
                    for idx2 in range(text_words_tod_emb_g_list[idx1].shape[0]):
                        sen_to_word.append([idx1, count_w])
                        if idx2 < text_words_tod_emb_g_list[idx1].shape[0] - 1:
                            word_to_word.append([count_w, count_w + 1])
                        count_w += 1
                if (len(word_to_word) == 0):
                    word_to_word.append([0, 0])


                cov_to_sen_audio = []
                for idx in range(len(audio_emb_g_list)):
                    cov_to_sen_audio.append([0, idx])
                sen_to_sen_audio = []
                for idx in range(len(audio_emb_g_list) - 1):
                    sen_to_sen_audio.append([idx, idx + 1])
                if (len(sen_to_sen_audio) == 0):
                    sen_to_sen_audio.append([0, 0])

                sen_to_word_audio = []
                count_w_audio = 0
                word_to_word_audio = []
                for idx1 in range(len(audio_emb_g_list)):
                    for idx2 in range(audio_words_emb_g_list[idx1].shape[0]):
                        sen_to_word_audio.append([idx1, count_w_audio])
                        if idx2 < audio_words_emb_g_list[idx1].shape[0] - 1:
                            word_to_word_audio.append([count_w_audio, count_w_audio + 1])
                        count_w_audio += 1
                if (len(word_to_word_audio) == 0):
                    word_to_word_audio.append([0, 0])

                graph_text = HeteroData()
                graph_text['coversation'], graph_text['sentence'], graph_text['word']
                graph_text["coversation", "connect", "sentence"]
                graph_text["sentence", "connect", "sentence"]
                graph_text["sentence", "connect", "word"]
                graph_text["word", "connect", "word"]


                graph_text['coversation'].x = text_summary_emb_g
                graph_text['sentence'].x = text_emb_tensor_g
                graph_text['word'].x = text_words_emb_tensor_g
                graph_text["coversation", "connect", "sentence"].edge_index = torch.tensor(
                    cov_to_sen).contiguous().transpose(-2, -1)
                if len(sen_to_sen) != 0:
                    graph_text["sentence", "connect", "sentence"].edge_index = torch.tensor(
                        sen_to_sen).contiguous().transpose(-2, -1)
                graph_text["sentence", "connect", "word"].edge_index = torch.tensor(sen_to_word).contiguous().transpose(
                    -2, -1)
                if len(word_to_word) != 0:
                    graph_text["word", "connect", "word"].edge_index = torch.tensor(
                        word_to_word).contiguous().transpose(-2, -1)

                graph_text = T.ToUndirected()(graph_text)

                graph_audio = HeteroData()
                graph_audio['coversation'], graph_audio['sentence'], graph_audio['word']
                graph_audio["coversation", "connect", "sentence"]
                graph_audio["sentence", "connect", "sentence"]
                graph_audio["sentence", "connect", "word"]
                graph_audio["word", "connect", "word"]

                graph_audio['coversation'].x = audio_conversational_emb_g
                graph_audio['sentence'].x = audio_emb_tensor_g
                graph_audio['word'].x = audio_words_emb_tensor_g
                graph_audio["coversation", "connect", "sentence"].edge_index = torch.tensor(
                    cov_to_sen_audio).contiguous().transpose(
                    -2, -1)
                if len(sen_to_sen_audio) != 0:
                    graph_audio["sentence", "connect", "sentence"].edge_index = torch.tensor(
                        sen_to_sen_audio).contiguous().transpose(-2,
                                                                 -1)
                graph_audio["sentence", "connect", "word"].edge_index = torch.tensor(
                    sen_to_word_audio).contiguous().transpose(-2, -1)
                if len(word_to_word_audio) != 0:
                    graph_audio["word", "connect", "word"].edge_index = torch.tensor(
                        word_to_word_audio).contiguous().transpose(-2, -1)

                graph_audio = T.ToUndirected()(graph_audio)

            cur_audio_path_g_re = waviemocap_cur_path_g + cur_spk_id + "-waviemocap_emb-" + basename + ".npy"
            cur_audio_sen_npy = np.load(cur_audio_path_g_re)
            cur_audio_sen_tensor = torch.tensor(cur_audio_sen_npy)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "attn_prior": attn_prior,
            "spker_embed": spker_embed,
            "emotion": emotion_id,
            "cur_sen_dig_text": text_emb_tensor_g,
            "cur_sen_dig_audio": audio_emb_tensor_g,
            "cur_audio_sen_an": cur_audio_sen_tensor,
            "text_positive_graph": text_positive_graph,
            "audio_positive_graph": audio_positive_graph,
            "text_negative_graph": text_negative_graph,
            "audio_negative_graph": audio_negative_graph,
            "cur_text_graph": graph_text,
            "cur_audio_graph": graph_audio
        }


        return sample

    def _clean_text(slef, text, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
            text = cleaner(text)
        return text

    def pad_history(self, 
            pad_size,
            history_text=None,
            history_text_emb=None,
            history_text_len=None,
            history_pitch=None,
            history_energy=None,
            history_duration=None,
            history_emotion=None,
            history_speaker=None,
            history_mel_len=None,
        ):
        for _ in range(pad_size):
            history_text.insert(0,np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_text_emb.insert(0,np.zeros(self.text_emb_size, dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.insert(0,0) if history_text_len is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_pitch.insert(0,np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.insert(0,np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.insert(0,np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.insert(0,0) if history_emotion is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_speaker.insert(0,0) if history_speaker is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_mel_len.insert(0,0) if history_mel_len is not None else None # meaningless zero padding, should be cut out by mask of history_len

    def pad_max_history(self, pad_list):
        max_len = max(t.shape[0] for t in pad_list)

        lengths = [t.shape[0] for t in pad_list]

        padded_list = [torch.nn.functional.pad(t, (0, 0, max_len - t.shape[0], 0), mode='constant', value=0)
                       if t.shape[0] < max_len else t for t in pad_list]

        return padded_list, lengths


    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []
            for line in f.readlines():
                if self.load_emotion:
                    n, s, t, r, e, *_ = line.strip("\n").split("|")
                else:
                    n, s, t, r, *_ = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if self.load_emotion:
                    emotion.append(e)
            return name, speaker, text, raw_text, emotion

    def reprocess(self, data, idxs):

        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs] if not self.learn_alignment else None
        attn_priors = [data[idx]["attn_prior"] for idx in idxs] if self.learn_alignment else None
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None
        emotions = np.array([data[idx]["emotion"] for idx in idxs]) if self.load_emotion else None

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        if self.learn_alignment:
            attn_priors = pad_3D(attn_priors, len(idxs), max(text_lens), max(mel_lens))
        else:
            durations = pad_1D(durations)


        text_positive_graphs = [data[idx]["text_positive_graph"] for idx in idxs]
        audio_positive_graphs = [data[idx]["audio_positive_graph"] for idx in idxs]
        text_negative_graphs = [data[idx]["text_negative_graph"] for idx in idxs]
        audio_negative_graphs = [data[idx]["audio_negative_graph"] for idx in idxs]


        cur_text_graphs = [data[idx]["cur_text_graph"] for idx in idxs]
        cur_audio_graphs = [data[idx]["cur_audio_graph"] for idx in idxs]

        cur_sen_dig_texts = [data[idx]["cur_sen_dig_text"] for idx in idxs]
        cur_sen_dig_audios = [data[idx]["cur_sen_dig_audio"] for idx in idxs]
        cur_audio_sen_ans = [data[idx]["cur_audio_sen_an"] for idx in idxs]

        cur_sen_dig_texts, cur_sen_dig_text_len  = self.pad_max_history(cur_sen_dig_texts)
        cur_sen_dig_audios, cur_sen_dig_audio_len = self.pad_max_history(cur_sen_dig_audios)

        cur_sen_dig_text_len = np.array(cur_sen_dig_text_len)
        cur_sen_dig_audio_len = np.array(cur_sen_dig_audio_len)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            emotions,
            cur_sen_dig_texts,
            cur_sen_dig_audios,
            cur_audio_sen_ans,
            cur_sen_dig_text_len,
            cur_sen_dig_audio_len,
            text_positive_graphs,
            audio_positive_graphs,
            text_negative_graphs,
            audio_negative_graphs,
            cur_text_graphs,
            cur_audio_graphs
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        t2 = time.time()

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.sub_dir_name = preprocess_config["path"]["sub_dir_name"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        self.load_emotion = model_config["multi_emotion"]
        self.history_type = model_config["history_encoder"]["type"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.basename, self.speaker, self.text, self.raw_text, self.emotion = self.process_meta(
            filepath
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)

        self.infer_path = "/xxx/300000/"

        # 控制N个数量
        self.N = 20

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[self.emotion[idx]] if self.load_emotion else None
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None



        if os.path.exists(self.infer_path + basename + ".pkl"):
            with open(self.infer_path + basename + ".pkl", 'rb') as f:
                seven_manba = pickle.load(f)
            text_posi_emb = seven_manba[0][:self.N]
            audio_posi_emb = seven_manba[1][:self.N]
        else:
            text_posi_emb = torch.zeros((50, 256), dtype=float)[:self.N]
            audio_posi_emb = torch.zeros((50, 256), dtype=float)[:self.N]



        # History
        dialog = basename.split("_")[2].strip("d")
        turn = int(basename.split("_")[0])
        history_len = min(self.max_history_len, turn)
        history_text = list()
        history_text_emb = list()
        history_text_len = list()
        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        history_mel_len = list()
        history = None
        if self.history_type != "none":
            history_basenames = sorted([tg_path.replace(".wav", "") for tg_path in os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
            history_basenames = history_basenames[:turn][-history_len:]



            graph_text = None
            graph_audio = None
            if turn != 0:
                text_path_g = self.preprocessed_path + "/text_emb/"
                text_words_tod_path_g = self.preprocessed_path + "/text_words_tod_emb/"
                waviemocap_path_g = self.preprocessed_path + "/audio_sencat_emb/"
                audio_word_path_g = self.preprocessed_path + "/audio_word_emb/"

                text_summary_path_g = self.preprocessed_path + "/text_summary_emb_iter/d" + dialog + "turn" + str(turn) + ".npy"
                text_summary_emb_g = np.load(text_summary_path_g)
                text_emb_g_list = []
                text_words_tod_emb_g_list = []
                audio_emb_g_list = []
                audio_words_emb_g_list = []


                for f_name in history_basenames:
                    f_list = f_name.split("_")


                    text_path_g_re = text_path_g + f_list[1] + "-text_emb-" + f_list[0] + "_" + f_list[1] + "_d" + dialog + ".npy"
                    text_emb_g = np.load(text_path_g_re)
                    text_emb_g_list.append(text_emb_g)
                    text_words_tod_path_g_re = text_words_tod_path_g + f_list[1] + "-text_words_tod_emb-" + f_list[0] + "_" + f_list[1] + "_d" + dialog + ".npy"
                    text_words_tod_emb_g = np.load(text_words_tod_path_g_re)
                    text_words_tod_emb_g_list.append(text_words_tod_emb_g)
                    waviemocap_path_g_re = waviemocap_path_g + f_list[1] + "-audio_sencat_emb-" + f_list[0] + "_" + f_list[1] + "_d" + dialog + ".npy"
                    audio_emb_g = np.load(waviemocap_path_g_re)
                    audio_emb_g_list.append(audio_emb_g)
                    audio_word_path_g_re = audio_word_path_g + f_list[1] + "-audio_word_emb-" + f_list[0] + "_" + f_list[1] + "_d" + dialog + ".npy"
                    audio_word_emb_g = np.load(audio_word_path_g_re)
                    audio_words_emb_g_list.append(audio_word_emb_g)

                f_list_cur = basename.split("_")
                text_path_g_re = text_path_g + f_list_cur[1] + "-text_emb-" + f_list_cur[0] + "_" + f_list_cur[1] + "_d" + dialog + ".npy"
                text_emb_g = np.load(text_path_g_re)
                text_emb_g_list.append(text_emb_g)
                text_words_tod_path_g_re = text_words_tod_path_g + f_list_cur[1] + "-text_words_tod_emb-" + f_list_cur[0] + "_" + f_list_cur[1] + "_d" + dialog + ".npy"
                text_words_tod_emb_g = np.load(text_words_tod_path_g_re)
                text_words_tod_emb_g_list.append(text_words_tod_emb_g)

                text_summary_emb_g = torch.tensor(text_summary_emb_g)
                text_summary_emb_g = torch.unsqueeze(text_summary_emb_g, dim=0)

                text_emb_tensor_g = np.stack(text_emb_g_list)
                text_emb_tensor_g = torch.tensor(text_emb_tensor_g)

                text_words_emb_tensor_g = np.concatenate(text_words_tod_emb_g_list, axis=0)
                text_words_emb_tensor_g = torch.tensor(text_words_emb_tensor_g)

                audio_emb_tensor_g = np.stack(audio_emb_g_list)
                audio_conversational_emb_g = np.mean(audio_emb_tensor_g, axis=0)
                audio_emb_tensor_g = torch.tensor(audio_emb_tensor_g)
                audio_conversational_emb_g = torch.tensor(audio_conversational_emb_g)
                audio_conversational_emb_g = torch.unsqueeze(audio_conversational_emb_g, dim=0)


                audio_words_emb_tensor_g = np.concatenate(audio_words_emb_g_list, axis=0)
                audio_words_emb_tensor_g = torch.tensor(audio_words_emb_tensor_g)
                audio_words_emb_tensor_g = torch.squeeze(audio_words_emb_tensor_g, dim=1)
                cov_to_sen = []
                for idx in range(len(text_emb_g_list)):
                    cov_to_sen.append([0, idx])
                sen_to_sen = []
                for idx in range(len(text_emb_g_list) - 1):
                    sen_to_sen.append([idx, idx + 1])
                sen_to_word = []
                count_w = 0
                word_to_word = []
                for idx1 in range(len(text_emb_g_list)):
                    for idx2 in range(text_words_tod_emb_g_list[idx1].shape[0]):
                        sen_to_word.append([idx1, count_w])
                        if idx2 < text_words_tod_emb_g_list[idx1].shape[0] - 1:
                            word_to_word.append([count_w, count_w + 1])
                        count_w += 1

                cov_to_sen_audio = []
                for idx in range(len(audio_emb_g_list)):
                    cov_to_sen_audio.append([0, idx])
                sen_to_sen_audio = []
                for idx in range(len(audio_emb_g_list) - 1):
                    sen_to_sen_audio.append([idx, idx + 1])
                sen_to_word_audio = []
                count_w_audio = 0
                word_to_word_audio = []
                for idx1 in range(len(audio_emb_g_list)):
                    for idx2 in range(audio_words_emb_g_list[idx1].shape[0]):
                        sen_to_word_audio.append([idx1, count_w_audio])
                        if idx2 < audio_words_emb_g_list[idx1].shape[0] - 1:
                            word_to_word_audio.append([count_w_audio, count_w_audio + 1])
                        count_w_audio += 1

                graph_text = HeteroData()
                graph_text['coversation'], graph_text['sentence'], graph_text['word']
                graph_text["coversation", "connect", "sentence"]
                graph_text["sentence", "connect", "sentence"]
                graph_text["sentence", "connect", "word"]
                graph_text["word", "connect", "word"]

                graph_text['coversation'].x = text_summary_emb_g
                graph_text['sentence'].x = text_emb_tensor_g
                graph_text['word'].x = text_words_emb_tensor_g
                graph_text["coversation", "connect", "sentence"].edge_index = torch.tensor(
                    cov_to_sen).contiguous().transpose(-2, -1)
                if len(sen_to_sen) != 0:
                    graph_text["sentence", "connect", "sentence"].edge_index = torch.tensor(
                        sen_to_sen).contiguous().transpose(-2, -1)
                graph_text["sentence", "connect", "word"].edge_index = torch.tensor(sen_to_word).contiguous().transpose(
                    -2, -1)
                if len(word_to_word) != 0:
                    graph_text["word", "connect", "word"].edge_index = torch.tensor(
                        word_to_word).contiguous().transpose(-2, -1)

                graph_text = T.ToUndirected()(graph_text)

                graph_audio = HeteroData()
                graph_audio['coversation'], graph_audio['sentence'], graph_audio['word']
                graph_audio["coversation", "connect", "sentence"]
                graph_audio["sentence", "connect", "sentence"]
                graph_audio["sentence", "connect", "word"]
                graph_audio["word", "connect", "word"]

                graph_audio['coversation'].x = audio_conversational_emb_g
                graph_audio['sentence'].x = audio_emb_tensor_g
                graph_audio['word'].x = audio_words_emb_tensor_g
                graph_audio["coversation", "connect", "sentence"].edge_index = torch.tensor(
                    cov_to_sen_audio).contiguous().transpose(
                    -2, -1)
                if len(sen_to_sen_audio) != 0:
                    graph_audio["sentence", "connect", "sentence"].edge_index = torch.tensor(
                        sen_to_sen_audio).contiguous().transpose(-2,
                                                           -1)
                graph_audio["sentence", "connect", "word"].edge_index = torch.tensor(
                    sen_to_word_audio).contiguous().transpose(-2, -1)
                if len(word_to_word_audio) != 0:
                    graph_audio["word", "connect", "word"].edge_index = torch.tensor(
                        word_to_word_audio).contiguous().transpose(-2, -1)

                graph_audio = T.ToUndirected()(graph_audio)


            if self.history_type == "Guo":
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename),
                )
                text_emb = np.load(text_emb_path)

                for i, h_basename in enumerate(history_basenames):
                    # h_idx = int(self.basename_to_id[h_basename])
                    h_speaker = h_basename.split("_")[1]
                    h_speaker_id = self.speaker_map[h_speaker]
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    h_text_emb = np.load(h_text_emb_path)
                    
                    history_text_emb.append(h_text_emb)
                    history_speaker.append(h_speaker_id)

                    # Padding
                    if i == history_len-1 and history_len < self.max_history_len:
                        self.pad_history(
                            self.max_history_len-history_len,
                            history_text_emb=history_text_emb,
                            history_speaker=history_speaker,
                        )
                if turn == 0:
                    self.pad_history(
                        self.max_history_len,
                        history_text_emb=history_text_emb,
                        history_speaker=history_speaker,
                    )

                history = {
                    "text_emb": text_emb,
                    "history_len": history_len,
                    "history_text_emb": history_text_emb,
                    "history_speaker": history_speaker,
                }

        return (basename, speaker_id, phone, raw_text, spker_embed, emotion_id, text_posi_emb, audio_posi_emb, graph_text, graph_audio, history)

    def pad_history(self, 
            pad_size,
            history_text=None,
            history_text_emb=None,
            history_text_len=None,
            history_pitch=None,
            history_energy=None,
            history_duration=None,
            history_emotion=None,
            history_speaker=None,
            history_mel_len=None,
        ):
        for _ in range(pad_size):
            history_text.insert(0,np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_text_emb.insert(0,np.zeros(self.text_emb_size, dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.insert(0,0) if history_text_len is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_pitch.insert(0,np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.insert(0,np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.insert(0,np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.insert(0,0) if history_emotion is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_speaker.insert(0,0) if history_speaker is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_mel_len.insert(0,0) if history_mel_len is not None else None # meaningless zero padding, should be cut out by mask of history_len

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []
            for line in f.readlines():
                if self.load_emotion:
                    n, s, t, r, e, *_ = line.strip("\n").split("|")
                else:
                    n, s, t, r, *_ = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if self.load_emotion:
                    emotion.append(e)
            return name, speaker, text, raw_text, emotion

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None
        emotions = np.array([d[5] for d in data]) if self.load_emotion else None

        text_posi_embs = [d[6] for d in data]
        audio_posi_embs = [d[7] for d in data]

        graph_texts = [d[8] for d in data]
        graph_audios = [d[9] for d in data]

        texts = pad_1D(texts)

        history_info = None
        if self.history_type != "none":
            if self.history_type == "Guo":
                text_embs = [d[10]["text_emb"] for d in data]
                history_lens = [d[10]["history_len"] for d in data]
                history_text_embs = [d[10]["history_text_emb"] for d in data]
                history_speakers = [d[10]["history_speaker"] for d in data]

                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_text_embs = np.array(history_text_embs)
                history_speakers = np.array(history_speakers)

                history_info = (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                )

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embeds, emotions, text_posi_embs, audio_posi_embs, graph_texts, graph_audios, history_info



class MAEDataset(Dataset):
    def __init__(
            self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.sub_dir_name = preprocess_config["path"]["sub_dir_name"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.load_spker_embed = model_config["multi_speaker"] \
                                and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        self.load_emotion = model_config["multi_emotion"]
        self.history_type = model_config["history_encoder"]["type"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.pitch_level_tag, self.energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)

        self.basename, self.speaker, self.text, self.raw_text, self.emotion = self.process_meta(
            filename
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        self.topk = train_config["p_n_sample"]["topK"]


        self.infer_path = "/xxx/300000/"

        self.N = xxx

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[self.emotion[idx]] if self.load_emotion else None
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel_{}".format(self.pitch_level_tag),
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch_{}".format(self.pitch_level_tag),
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy_{}".format(self.energy_level_tag),
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        if self.learn_alignment:
            attn_prior_path = os.path.join(
                self.preprocessed_path,
                "attn_prior",
                "{}-attn_prior-{}.npy".format(speaker, basename),
            )
            attn_prior = np.load(attn_prior_path)
            duration = None
        else:
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)
            attn_prior = None
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None


        conversation_index = basename.split('_')[-1]

        text_positive_preprocessed_path = self.preprocessed_path + "/train_heterogeneous_graph_Topk/text_positive_graph/"

        audio_positive_preprocessed_path = self.preprocessed_path + "/train_heterogeneous_graph_Topk/audio_positive_graph/"

        text_negative_preprocessed_path = self.preprocessed_path + "/train_heterogeneous_graph_Topk/text_negative_graph/"

        audio_negative_preprocessed_path = self.preprocessed_path + "/train_heterogeneous_graph_Topk/audio_negative_graph/"

        with open(text_positive_preprocessed_path + conversation_index + ".pkl", 'rb') as f:
            text_positive_graph = pickle.load(f)
            text_positive_graph = text_positive_graph[:self.topk]

        with open(audio_positive_preprocessed_path + conversation_index + ".pkl", 'rb') as f:
            audio_positive_graph = pickle.load(f)
            audio_positive_graph = audio_positive_graph[:self.topk]

        with open(text_negative_preprocessed_path + conversation_index + ".pkl", 'rb') as f:
            text_negative_graph = pickle.load(f)
            text_negative_graph = text_negative_graph[:self.topk]

        with open(audio_negative_preprocessed_path + conversation_index + ".pkl", 'rb') as f:
            audio_negative_graph = pickle.load(f)
            audio_negative_graph = audio_negative_graph[:self.topk]

        # History
        dialog = basename.split("_")[2].strip("d")
        turn = int(basename.split("_")[0])
        history_len = min(self.max_history_len, turn)
        history_text = list()
        history_text_emb = list()
        history_text_len = list()
        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        history_mel_len = list()
        history = None

        if self.history_type != "none":
            history_basenames = sorted([tg_path.replace(".wav", "") for tg_path in
                                        os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if
                                        ".wav" in tg_path], key=lambda x: int(x.split("_")[0]))
            history_basenames = history_basenames[:turn][-history_len:]

            graph_text = None
            graph_audio = None
            if turn != 0:
                text_path_g = self.preprocessed_path + "/text_emb/"
                text_words_tod_path_g = self.preprocessed_path + "/text_words_tod_emb/"
                waviemocap_path_g = self.preprocessed_path + "/audio_sencat_emb/"
                audio_word_path_g = self.preprocessed_path + "/audio_word_emb/"

                # dialogue = ''
                text_summary_path_g = self.preprocessed_path + "/text_summary_emb_iter/d" + dialog + "turn" + str(
                    turn) + ".npy"
                text_summary_emb_g = np.load(text_summary_path_g)
                text_emb_g_list = []
                text_words_tod_emb_g_list = []
                audio_emb_g_list = []
                audio_words_emb_g_list = []

                for f_name in history_basenames:
                    f_list = f_name.split("_")

                    text_path_g_re = text_path_g + f_list[1] + "-text_emb-" + f_list[0] + "_" + f_list[
                        1] + "_d" + dialog + ".npy"
                    text_emb_g = np.load(text_path_g_re)
                    text_emb_g_list.append(text_emb_g)
                    text_words_tod_path_g_re = text_words_tod_path_g + f_list[1] + "-text_words_tod_emb-" + f_list[
                        0] + "_" + f_list[1] + "_d" + dialog + ".npy"
                    text_words_tod_emb_g = np.load(text_words_tod_path_g_re)
                    text_words_tod_emb_g_list.append(text_words_tod_emb_g)
                    waviemocap_path_g_re = waviemocap_path_g + f_list[1] + "-audio_sencat_emb-" + f_list[0] + "_" + \
                                           f_list[1] + "_d" + dialog + ".npy"
                    audio_emb_g = np.load(waviemocap_path_g_re)
                    audio_emb_g_list.append(audio_emb_g)
                    audio_word_path_g_re = audio_word_path_g + f_list[1] + "-audio_word_emb-" + f_list[0] + "_" + \
                                           f_list[1] + "_d" + dialog + ".npy"
                    audio_word_emb_g = np.load(audio_word_path_g_re)
                    audio_words_emb_g_list.append(audio_word_emb_g)

                f_list_cur = basename.split("_")
                text_path_g_re = text_path_g + f_list_cur[1] + "-text_emb-" + f_list_cur[0] + "_" + f_list_cur[
                    1] + "_d" + dialog + ".npy"
                text_emb_g = np.load(text_path_g_re)
                text_emb_g_list.append(text_emb_g)
                text_words_tod_path_g_re = text_words_tod_path_g + f_list_cur[1] + "-text_words_tod_emb-" + f_list_cur[
                    0] + "_" + f_list_cur[1] + "_d" + dialog + ".npy"
                text_words_tod_emb_g = np.load(text_words_tod_path_g_re)
                text_words_tod_emb_g_list.append(text_words_tod_emb_g)


                text_summary_emb_g = torch.tensor(text_summary_emb_g)
                text_summary_emb_g = torch.unsqueeze(text_summary_emb_g, dim=0)

                text_emb_tensor_g = np.stack(text_emb_g_list)
                text_emb_tensor_g = torch.tensor(text_emb_tensor_g)

                text_words_emb_tensor_g = np.concatenate(text_words_tod_emb_g_list, axis=0)
                text_words_emb_tensor_g = torch.tensor(text_words_emb_tensor_g)

                audio_emb_tensor_g = np.stack(audio_emb_g_list)
                audio_conversational_emb_g = np.mean(audio_emb_tensor_g, axis=0)
                audio_emb_tensor_g = torch.tensor(audio_emb_tensor_g)
                audio_conversational_emb_g = torch.tensor(audio_conversational_emb_g)
                audio_conversational_emb_g = torch.unsqueeze(audio_conversational_emb_g, dim=0)

                audio_words_emb_tensor_g = np.concatenate(audio_words_emb_g_list, axis=0)
                audio_words_emb_tensor_g = torch.tensor(audio_words_emb_tensor_g)
                audio_words_emb_tensor_g = torch.squeeze(audio_words_emb_tensor_g, dim=1)
                cov_to_sen = []
                for idx in range(len(text_emb_g_list)):
                    cov_to_sen.append([0, idx])
                sen_to_sen = []
                for idx in range(len(text_emb_g_list) - 1):
                    sen_to_sen.append([idx, idx + 1])
                sen_to_word = []
                count_w = 0
                word_to_word = []
                for idx1 in range(len(text_emb_g_list)):
                    for idx2 in range(text_words_tod_emb_g_list[idx1].shape[0]):
                        sen_to_word.append([idx1, count_w])
                        if idx2 < text_words_tod_emb_g_list[idx1].shape[0] - 1:
                            word_to_word.append([count_w, count_w + 1])
                        count_w += 1

                cov_to_sen_audio = []
                for idx in range(len(audio_emb_g_list)):
                    cov_to_sen_audio.append([0, idx])
                sen_to_sen_audio = []
                for idx in range(len(audio_emb_g_list) - 1):
                    sen_to_sen_audio.append([idx, idx + 1])
                sen_to_word_audio = []
                count_w_audio = 0
                word_to_word_audio = []
                for idx1 in range(len(audio_emb_g_list)):
                    for idx2 in range(audio_words_emb_g_list[idx1].shape[0]):
                        sen_to_word_audio.append([idx1, count_w_audio])
                        if idx2 < audio_words_emb_g_list[idx1].shape[0] - 1:
                            word_to_word_audio.append([count_w_audio, count_w_audio + 1])
                        count_w_audio += 1

                graph_text = HeteroData()
                graph_text['coversation'], graph_text['sentence'], graph_text['word']
                graph_text["coversation", "connect", "sentence"]
                graph_text["sentence", "connect", "sentence"]
                graph_text["sentence", "connect", "word"]
                graph_text["word", "connect", "word"]

                graph_text['coversation'].x = text_summary_emb_g
                graph_text['sentence'].x = text_emb_tensor_g
                graph_text['word'].x = text_words_emb_tensor_g
                graph_text["coversation", "connect", "sentence"].edge_index = torch.tensor(
                    cov_to_sen).contiguous().transpose(-2, -1)
                if len(sen_to_sen) != 0:
                    graph_text["sentence", "connect", "sentence"].edge_index = torch.tensor(
                        sen_to_sen).contiguous().transpose(-2, -1)
                graph_text["sentence", "connect", "word"].edge_index = torch.tensor(sen_to_word).contiguous().transpose(
                    -2, -1)
                if len(word_to_word) != 0:
                    graph_text["word", "connect", "word"].edge_index = torch.tensor(
                        word_to_word).contiguous().transpose(-2, -1)

                graph_text = T.ToUndirected()(graph_text)

                graph_audio = HeteroData()
                graph_audio['coversation'], graph_audio['sentence'], graph_audio['word']
                graph_audio["coversation", "connect", "sentence"]
                graph_audio["sentence", "connect", "sentence"]
                graph_audio["sentence", "connect", "word"]
                graph_audio["word", "connect", "word"]

                graph_audio['coversation'].x = audio_conversational_emb_g
                graph_audio['sentence'].x = audio_emb_tensor_g
                graph_audio['word'].x = audio_words_emb_tensor_g
                graph_audio["coversation", "connect", "sentence"].edge_index = torch.tensor(
                    cov_to_sen_audio).contiguous().transpose(
                    -2, -1)
                if len(sen_to_sen_audio) != 0:
                    graph_audio["sentence", "connect", "sentence"].edge_index = torch.tensor(
                        sen_to_sen_audio).contiguous().transpose(-2,
                                                                 -1)
                graph_audio["sentence", "connect", "word"].edge_index = torch.tensor(
                    sen_to_word_audio).contiguous().transpose(-2, -1)
                if len(word_to_word_audio) != 0:
                    graph_audio["word", "connect", "word"].edge_index = torch.tensor(
                        word_to_word_audio).contiguous().transpose(-2, -1)

                graph_audio = T.ToUndirected()(graph_audio)


            if self.history_type == "Guo":
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename),
                )
                text_emb = np.load(text_emb_path)

                for i, h_basename in enumerate(history_basenames):
                    h_speaker = h_basename.split("_")[1]
                    h_speaker_id = self.speaker_map[h_speaker]
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    h_text_emb = np.load(h_text_emb_path)

                    history_text_emb.append(h_text_emb)
                    history_speaker.append(h_speaker_id)

                    # Padding
                    if i == history_len - 1 and history_len < self.max_history_len:
                        self.pad_history(
                            self.max_history_len - history_len,
                            history_text_emb=history_text_emb,
                            history_speaker=history_speaker,
                        )
                if turn == 0:
                    self.pad_history(
                        self.max_history_len,
                        history_text_emb=history_text_emb,
                        history_speaker=history_speaker,
                    )

                history = {
                    "text_emb": text_emb,
                    "history_len": history_len,
                    "history_text_emb": history_text_emb,
                    "history_speaker": history_speaker,
                }



        if os.path.exists(self.infer_path + basename + ".pkl"):
            with open(self.infer_path + basename + ".pkl", 'rb') as f:
                seven_manba = pickle.load(f)
            text_posi_emb = seven_manba[0][:self.N]
            audio_posi_emb = seven_manba[1][:self.N]
        else:
            text_posi_emb = torch.zeros((50, 256), dtype=float)[:self.N]
            audio_posi_emb = torch.zeros((50, 256), dtype=float)[:self.N]

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "attn_prior": attn_prior,
            "spker_embed": spker_embed,
            "emotion": emotion_id,
            "history": history,
            "text_positive_graph": text_positive_graph,
            "audio_positive_graph": audio_positive_graph,
            "text_negative_graph": text_negative_graph,
            "audio_negative_graph": audio_negative_graph,
            "cur_text_graph": graph_text,
            "cur_audio_graph": graph_audio,
            "text_posi_emb": text_posi_emb,
            "audio_posi_emb": audio_posi_emb,

        }

        return sample

    def _clean_text(slef, text, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
            text = cleaner(text)
        return text

    def pad_history(self,
                    pad_size,
                    history_text=None,
                    history_text_emb=None,
                    history_text_len=None,
                    history_pitch=None,
                    history_energy=None,
                    history_duration=None,
                    history_emotion=None,
                    history_speaker=None,
                    history_mel_len=None,
                    ):
        for _ in range(pad_size):
            history_text.insert(0, np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_text_emb.insert(0, np.zeros(self.text_emb_size,
                                                dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.insert(0,
                                    0) if history_text_len is not None else None  # meaningless zero padding, should be cut out by mask of history_len
            history_pitch.insert(0, np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.insert(0, np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.insert(0, np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.insert(0,
                                   0) if history_emotion is not None else None  # meaningless zero padding, should be cut out by mask of history_len
            history_speaker.insert(0,
                                   0) if history_speaker is not None else None  # meaningless zero padding, should be cut out by mask of history_len
            history_mel_len.insert(0,
                                   0) if history_mel_len is not None else None  # meaningless zero padding, should be cut out by mask of history_len

    def process_meta(self, filename):
        with open(
                os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []
            for line in f.readlines():
                if self.load_emotion:
                    n, s, t, r, e, *_ = line.strip("\n").split("|")
                else:
                    n, s, t, r, *_ = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if self.load_emotion:
                    emotion.append(e)
            return name, speaker, text, raw_text, emotion

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs] if not self.learn_alignment else None
        attn_priors = [data[idx]["attn_prior"] for idx in idxs] if self.learn_alignment else None
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None
        emotions = np.array([data[idx]["emotion"] for idx in idxs]) if self.load_emotion else None

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        if self.learn_alignment:
            attn_priors = pad_3D(attn_priors, len(idxs), max(text_lens), max(mel_lens))
        else:
            durations = pad_1D(durations)

        text_positive_graphs = [data[idx]["text_positive_graph"] for idx in idxs]
        audio_positive_graphs = [data[idx]["audio_positive_graph"] for idx in idxs]
        text_negative_graphs = [data[idx]["text_negative_graph"] for idx in idxs]
        audio_negative_graphs = [data[idx]["audio_negative_graph"] for idx in idxs]
        cur_text_graphs = [data[idx]["cur_text_graph"] for idx in idxs]
        cur_audio_graphs = [data[idx]["cur_audio_graph"] for idx in idxs]

        history_info = None
        if self.history_type != "none":
            if self.history_type == "Guo":
                text_embs = [data[idx]["history"]["text_emb"] for idx in idxs]
                history_lens = [data[idx]["history"]["history_len"] for idx in idxs]
                history_text_embs = [data[idx]["history"]["history_text_emb"] for idx in idxs]
                history_speakers = [data[idx]["history"]["history_speaker"] for idx in idxs]

                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_text_embs = np.array(history_text_embs)
                history_speakers = np.array(history_speakers)

                history_info = (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                )

        text_posi_embs =  [data[idx]["text_posi_emb"] for idx in idxs]
        audio_posi_embs = [data[idx]["audio_posi_emb"] for idx in idxs]


        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            emotions,
            history_info,
            text_positive_graphs,
            audio_positive_graphs,
            text_negative_graphs,
            audio_negative_graphs,
            cur_text_graphs,
            cur_audio_graphs,
            text_posi_embs,
            audio_posi_embs
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
