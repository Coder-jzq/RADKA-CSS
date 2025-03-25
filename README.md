# Retrieval-Augmented Dialogue Knowledge Aggregation for Expressive Conversational Speech Synthesis (RADKA-CSS)




## Introduction

This is an implementation of the following paper. ["Retrieval-Augmented Dialogue Knowledge Aggregation for Expressive Conversational Speech Synthesis"](https://www.sciencedirect.com/science/article/pii/S1566253525000211) (Accepted by Information Fusion 2025)

[Rui Liu](https://ttslr.github.io/people.html), [**Zhenqi Jia**](https://coder-jzq.github.io/), Feilong Bao, Haizhou Li




## Demo Page

[Speech Demo](https://coder-jzq.github.io/RADKA-CSS-Website/)




## Dataset

You can download the [dataset](https://drive.google.com/drive/folders/1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL) from DailyTalk.




## Pre-trained models

The Hugging Face URL of Sentence-BERT:  https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1

The Hugging Face URL of Wav2Vec2-IEMOCAP: https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP

The Hugging Face URL of X-vectors: https://huggingface.co/speechbrain/spkrec-xvect-voxceleb

The Hugging Face URL of TODBERT: https://huggingface.co/TODBERT/TOD-BERT-JNT-V1

The Hugging Face URL of Wav2Vec2.0: https://huggingface.co/facebook/wav2vec2-base-960h

The Hugging Face URL of Dialogue Text Summarization Extraction (bart-largecnn-samsum): https://huggingface.co/philschmid/bart-large-cnn-samsum




## Preprocessing

Run

> python3 prepare_align.py --dataset DailyTalk

for some preparations.

For the forced alignment, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences. Pre-extracted alignments for the datasets are provided [here](https://drive.google.com/drive/folders/1fizpyOiQ1lG2UDaMlXnT3Ll4_j6Xwg7K?usp=sharing). You have to unzip the files in `preprocessed_data/DailyTalk/TextGrid/`. Alternately, you can [run the aligner by yourself](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/index.html). Please note that our pretrained models are not trained with supervised duration modeling (they are trained with `learn_alignment: True`).

After that, run the preprocessing script by

> python3 preprocess.py --dataset DailyTalk




## Training

Train RADKA-CSS with

> python3 train.py --dataset DailyTalk




## Inference

Only the batch inference is supported as the generation of a turn may need contextual history of the conversation. Try

> python3 synthesize.py --source preprocessed_data/DailyTalk/test_*.txt --restore_step RESTORE_STEP --mode batch --dataset DailyTalk

to synthesize all utterances in `preprocessed_data/DailyTalk/test_*.txt`.




## Citation

If you would like to use our dataset and code or refer to our paper, please cite as follows.
```bash
@article{LIU2025102948,
title = {Retrieval-Augmented Dialogue Knowledge Aggregation for expressive conversational speech synthesis},
journal = {Information Fusion},
volume = {118},
pages = {102948},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.102948},
url = {https://www.sciencedirect.com/science/article/pii/S1566253525000211},
author = {Rui Liu and Zhenqi Jia and Feilong Bao and Haizhou Li},
keywords = {Conversational speech synthesis, Retrieval-augmented generation, Multi-source style knowledge, Multi-granularity, Heterogeneous graph},
abstract = {Conversational speech synthesis (CSS) aims to take the current dialogue (CD) history as a reference to synthesize expressive speech that aligns with the conversational style. Unlike CD, stored dialogue (SD) contains preserved dialogue fragments from earlier stages of user–agent interaction, which include style expression knowledge relevant to scenarios similar to those in CD. Note that this knowledge plays a significant role in enabling the agent to synthesize expressive conversational speech that generates empathetic feedback. However, prior research has overlooked this aspect. To address this issue, we propose a novel Retrieval-Augmented Dialogue Knowledge Aggregation scheme for expressive CSS, termed RADKA-CSS, which includes three main components: (1) To effectively retrieve dialogues from SD that are similar to CD in terms of both semantic and style. First, we build a stored dialogue semantic-style database (SDSSD) which includes the text and audio samples. Then, we design a multi-attribute retrieval scheme to match the dialogue semantic and style vectors of the CD with the stored dialogue semantic and style vectors in the SDSSD, retrieving the most similar dialogues. (2) To effectively utilize the style knowledge from CD and SD, we propose adopting the multi-granularity graph structure to encode the dialogue and introducing a multi-source style knowledge aggregation mechanism. (3) Finally, the aggregated style knowledge are fed into the speech synthesizer to help the agent synthesize expressive speech that aligns with the conversational style. We conducted a comprehensive and in-depth experiment based on the DailyTalk dataset, which is a benchmarking dataset for the CSS task. Both objective and subjective evaluations demonstrate that RADKA-CSS outperforms baseline models in expressiveness rendering. Code and audio samples can be found at: https://github.com/Coder-jzq/RADKA-CSS.}
}

```




## Contact the Author

E-mail：jiazhenqi7@163.com

Homepage: https://coder-jzq.github.io/

S2LAB Homepage: https://ttslr.github.io/

