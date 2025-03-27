# Awesome Darija Arabic NLP Resources [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img src="https://img.shields.io/badge/Darija-NLP-green" alt="Darija NLP"/>
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License"/>
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs Welcome"/>
</p>

<p align="center">
  A curated list of resources for Natural Language Processing (NLP) in Moroccan Darija Arabic.
</p>

## 📋 Contents

- [Datasets](#-datasets)
  - [Text Datasets](#text-datasets)
  - [Speech Datasets](#speech-datasets)
  - [Multimodal Datasets](#multimodal-datasets)
- [Models](#-models)
  - [Language Models](#language-models)
  - [Translation Models](#translation-models)
  - [Speech Models](#speech-models)
  - [NER & Classification Models](#ner--classification-models)
  - [Summarization Models](#summarization-models)
  - [Frameworks & Tools](#frameworks--tools)
- [Benchmarks](#-benchmarks)
  - [General Language Understanding](#general-language-understanding)
  - [Translation Benchmarks](#translation-benchmarks)
  - [Summarization Benchmarks](#summarization-benchmarks)
  - [LLM Evaluation Frameworks](#llm-evaluation-frameworks)
- [Tools & Libraries](#-tools--libraries)
- [Learning Resources](#-learning-resources)
- [Research Papers](#-research-papers)
- [Contributing](#-contributing)
- [License](#-license)

## 📊 Datasets

### Text Datasets

#### Translation & Lexical Resources

| Dataset | Description | Size | Year | Link |
|---------|-------------|------|------|------|
| **Darija Open Dataset (DODa)** | The largest open-source collaborative project for Darija-English translation. Includes words with different spellings, verb-to-noun and masculine-to-feminine correspondences, conjugations, and translated sentences. Available in both Latin and Arabic alphabets. | ~150,000 entries | 2021 | [GitHub](https://github.com/darija-open-dataset/dataset) / [Website](https://darija-open-dataset.github.io/) |
| **Moroccan Dialect Darija Open Dataset** | Open-source contributions in Darija. | >13K entries | 2021 | [Website](https://darija-open-dataset.github.io/) |
| **darija_english** | Darija-English pairs for translation tasks. | 10k-45.1k rows | - | [Hugging Face](https://huggingface.co/datasets/atlasia/darija_english) |
| **Moroccan Darija Wikipedia dataset** | Dataset derived from Moroccan Darija Wikipedia. | - | - | [Hugging Face](https://huggingface.co/datasets/AbderrahmanSkiredj1/moroccan_darija_wikipedia_dataset) |

#### News & Media

| Dataset | Description | Size | Year | Link |
|---------|-------------|------|------|------|
| **Goud.ma** | News dataset for summarization in Darija. | 158k articles | 2022 | [GitHub](https://github.com/issam9/goud-summarization-dataset) / [Hugging Face](https://huggingface.co/datasets/Goud/Goud-sum) |
| **MNAD (Moroccan News Articles Dataset)** | Collection of documents from Moroccan news websites. | 418,563 documents | 2021 | [Kaggle](https://www.kaggle.com/datasets/jmourad100/mnad-moroccan-news-articles-dataset) / [Hugging Face](https://huggingface.co/datasets/J-Mourad/MNAD.v2) |

#### Sentiment Analysis & Content Classification

| Dataset | Description | Size | Year | Link |
|---------|-------------|------|------|------|
| **Moroccan Arabic Sentiment Analysis Corpus** | Twitter dataset for sentiment analysis. | 2,000 entries | 2018 | [GitHub](https://github.com/ososs/Arabic-Sentiment-Analysis-corpus/blob/master/MSAC.arff) |
| **OMCD (Offensive Moroccan Comments Dataset)** | YouTube comments labeled for offensive content. | 8,024 comments | 2023 | [GitHub](https://github.com/kabilessefar/OMCD-Offensive-Moroccan-Comments-Dataset) |
| **Moroccan Darija Offensive Language Detection Dataset** | Sentences labeled for offensive language in Darija. | 20,402 sentences | - | [Mendeley](https://data.mendeley.com/datasets/2y4m97b7dc/3) |
| **Moroccan Arabic Corpus (MAC)** | Large Moroccan corpus for sentiment analysis. | - | - | [HAL](https://hal.science/hal-03670346) |
| **darija-arabic-classification** | Records for dialect classification (85% Darija, 15% MSA). | 1.5k records | - | [Hugging Face](https://huggingface.co/datasets/sawalni-ai/darija-arabic-classification) |
| **DarijaBanking** | Banking intent detection dataset with queries in Arabic/Darija. | >7,200 queries | - | [Hugging Face](https://huggingface.co/datasets/AbderrahmanSkiredj1/DarijaBanking) |

#### Dialect Identification

| Dataset | Description | Size | Year | Link |
|---------|-------------|------|------|------|
| **IADD (Integrated Arabic Dialect Identification Dataset)** | Texts from various sources including Maghrebi (Moroccan). | 135,804 texts | 2022 | [GitHub](https://github.com/Jihadz/IADD) |
| **QADI (QCRI Arabic Dialect Identification)** | Twitter dataset including Maghrebi (Moroccan). | 540k tweets | 2020 | [GitHub](https://github.com/qcri/QADI) |
| **Dialectal Arabic Datasets** | Twitter dataset with tweets from different Arabic dialects. | 350 tweets per region | 2018 | [GitHub](https://github.com/qcri/dialectal_arabic_resources) |
| **arabic_pos_dialect** | POS tagging dataset for Arabic dialects, including Maghrebi. | 350 rows for Maghrebi | - | [Hugging Face](https://huggingface.co/datasets/QCRI/arabic_pos_dialect) |
| **ADI17** | Fine-grained Arabic dialect identification dataset. | - | - | [ResearchGate](https://www.researchgate.net/publication/338843159_ADI17_A_Fine-Grained_Arabic_Dialect_Identification_Dataset) |
| **MADAR** | Multi-Arabic dialect applications and resources. | - | - | [Website](https://sites.google.com/nyu.edu/madar/) |

#### Named Entity Recognition

| Dataset | Description | Size | Year | Link |
|---------|-------------|------|------|------|
| **DarNERcorp** | Manually annotated corpus for Named Entity Recognition in Darija. | 65,905 tokens | 2023 | [Mendeley](https://data.mendeley.com/datasets/286sss4k9v/4) |

#### General Text Corpora

| Dataset | Description | Size | Year | Link |
|---------|-------------|------|------|------|
| **Darija-Stories-Dataset** | Large collection of Darija stories. | >70M tokens | - | [Hugging Face](https://huggingface.co/datasets/Ali-C137/Darija-Stories-Dataset) |
| **Darija-SFT-Mixture** | Dataset for fine-tuning LLMs with instruction samples in Darija. | 458K samples | - | [Hugging Face](https://huggingface.co/datasets/MBZUAI-Paris/Darija-SFT-Mixture) |
| **MSDA Open Datasets** | Social media posts in Arabic dialects, including Darija. | - | 2020 | [Website](https://msda.um6p.ma/msda_datasets) |
| **DART** | Dataset including Maghrebi, Egyptian, Levantine, Gulf, and Iraqi Arabic. | - | - | [QSpace](https://qspace.qu.edu.qa/handle/10576/15265) |

### Speech Datasets

| Dataset | Description | Size | Year | Link |
|---------|-------------|------|------|------|
| **Dvoice** | ASR dataset for Moroccan dialectal Arabic. | 2,392 training, 600 testing files | 2021 | [GitHub](https://github.com/AIOXLABS/DVoice) |
| **darija-stt-mix** | Speech recognition dataset for Darija. | 13,178 rows | - | [Hugging Face](https://huggingface.co/datasets/ayoubkirouane/darija-stt-mix) |
| **DARIJA-C Corpus** | Crowdsourced speech corpus for translating spoken Darija to MSA text. | 50 hours (18,000 recordings) | - | [Paper](https://ijettjournal.org/Volume-72/Issue-10/IJETT-V72I10P125.pdf) |
| **Moroccan Darija Wiki Audio Dataset** | Parallel text and speech samples from Wikipedia Darija. | 551 samples | - | [Hugging Face](https://huggingface.co/datasets/atlasia/Moroccan-Darija-Wiki-Audio-Dataset) |

### Multimodal Datasets

| Dataset | Description | Size | Year | Link |
|---------|-------------|------|------|------|
| **ASAYAR** | Images from Moroccan highways, annotated for scene text localization. | 1,763 images | 2020 | [Website](https://vcar.github.io/ASAYAR/) |
| **MORED** | Moroccan buildings' electricity consumption dataset. | - | 2020 | [GitHub](https://github.com/MOREDataset/MORED) |

## 🤖 Models

### Language Models

| Model | Description | Architecture | Year | Link |
|-------|-------------|--------------|------|------|
| **Atlas-Chat** | First family of LLMs specifically designed for Moroccan Arabic. Available in 2B and 9B parameter versions. Outperforms existing Arabic-focused models in Darija-specific tasks. | Decoder-only (based on Google Gemma 2) | 2023 | [Hugging Face (7B)](https://huggingface.co/MBZUAI-Paris/Atlas-Chat-7B) / [GGUF Version](https://huggingface.co/QuantFactory/Atlas-Chat-9B-GGUF) |
| **Darija-GPT** | Generative text model for Algerian Darija (closely related to Moroccan Darija). | GPT-2 architecture | - | [GitHub](https://github.com/Kirouane-Ayoub/Darija-GPT) |
| **DarijaBERT** | Encoder-only model for Darija, fine-tuned for various NLP tasks. | BERT-based | - | [GitHub](https://github.com/AIOXLABS/DBert) |
| **DarijaBERT-arabizi** | BERT model for Darija written in Latin script (Arabizi). | BERT-based | - | [Hugging Face](https://huggingface.co/SI2M-Lab/DarijaBERT-arabizi) |
| **DarijaBERT-mix** | BERT model for Darija with mixed script support. | BERT-based | - | [Hugging Face](https://huggingface.co/SI2M-Lab/DarijaBERT-mix) |
| **MorRoBERTa** | RoBERTa-based model for Moroccan Arabic. | RoBERTa | - | [Hugging Face](https://huggingface.co/otmangi/MorRoBERTa) |
| **MorrBERT** | BERT-based model for Moroccan Arabic. | BERT | - | [Hugging Face](https://huggingface.co/otmangi/MorrBERT) |
| **MARBERT** | Large-scale pre-trained masked language model for Arabic dialects. | BERT-based | - | [Hugging Face](https://huggingface.co/UBC-NLP/MARBERTv2) |
| **AraGPT2** | GPT-2 model pre-trained on Arabic text. | GPT-2 | - | [Hugging Face](https://huggingface.co/aubmindlab/aragpt2-base) |
| **AraBERT** | BERT model pre-trained on Arabic text. | BERT | - | [Hugging Face](https://huggingface.co/aubmindlab/bert-base-arabertv2) |

### Translation Models

| Model | Description | Architecture | Year | Link |
|-------|-------------|--------------|------|------|
| **English-Darija Transformer** | Neural machine translation model for English-Darija translation. Fine-tuned on 15,000+ translation pairs from DODa. | Transformer (based on Helsinki-NLP/opus-mt-tc-big-en-ar) | - | [LinkedIn Article](https://www.linkedin.com/pulse/journey-translating-english-moroccan-darija-natural-language-lachkar-qx7pe) |
| **AdabTranslate Darija** | Translation model between Darija and Modern Standard Arabic (MSA). Trained on 26,000 text pairs. | Transformer | - | [Hugging Face](https://huggingface.co/itsmeussa/AdabTranslate-Darija) |
| **AraT5 Darija to MSA** | T5-based model for translating Darija to Modern Standard Arabic. | T5 | - | [Hugging Face](https://huggingface.co/Saidtaoussi/AraT5_Darija_to_MSA) |
| **Seamless Darija-English** | Translation model between Darija and English. | Transformer | - | [Hugging Face](https://huggingface.co/AnasAber/seamless-darija-eng) |

### Speech Models

| Model | Description | Architecture | Year | Link |
|-------|-------------|--------------|------|------|
| **DVoice Darija ASR** | Automatic Speech Recognition model for Moroccan Darija. Fine-tuned on the DVoice dataset. | wav2vec 2.0 | 2021 | [Hugging Face](https://huggingface.co/aioxlabs/asr-wav2vec2-dvoice-darija) / [PromptLayer](https://www.promptlayer.com/models/asr-wav2vec2-dvoice-darija) |

### NER & Classification Models

| Model | Description | Architecture | Year | Link |
|-------|-------------|--------------|------|------|
| **darija-ner** | Named Entity Recognition model for Darija. | BERT-based | - | [Hugging Face](https://huggingface.co/hananour/darija-ner) |
| **magbert-ner** | Named Entity Recognition model for Maghrebi Arabic. | BERT-based | - | [Hugging Face](https://huggingface.co/TypicaAI/magbert-ner) |
| **BERTouch** | Banking intent detection model for Darija. | BERT-based (XLM-RoBERTa) | - | [Hugging Face](https://huggingface.co/AbderrahmanSkiredj1/BERTouch) |
| **CAMeLBERT-DA-Sentiment** | Sentiment analysis model for Arabic dialects. | BERT-based | - | [Hugging Face](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment) |

### Summarization Models

| Model | Description | Architecture | Year | Link |
|-------|-------------|--------------|------|------|
| **T5 Darija Summarization** | Text summarization model for Darija. | T5 | - | [Hugging Face](https://huggingface.co/Kamel/t5-darija-summarization) |
| **AraBERT Summarization Goud** | Summarization model fine-tuned on Goud.ma dataset. | BERT-based | - | [Hugging Face](https://huggingface.co/Goud/AraBERT-summarization-goud) |
| **MArSum** | Summarization models for Moroccan Arabic. | Transformer-based | - | [GitHub](https://github.com/KamelGaanoun/MoroccanSummarization) |

### Frameworks & Tools

| Tool | Description | Type | Year | Link |
|------|-------------|------|------|------|
| **CAMeL Tools** | Suite of Arabic NLP tools with Darija support for POS tagging, NER, etc. | Various (rule-based, ML) | - | [GitHub](https://github.com/CAMeL-Lab/camel_tools) |
| **Farasa** | Package for Arabic language processing with Darija support. | Rule-based and ML-based | - | [Website](https://farasa.qcri.org/) |
| **SAFAR** | Framework for Arabic language processing with Darija support. | Monolingual framework | - | [Website](http://arabic.emi.ac.ma/safar/) |

## 📏 Benchmarks

### General Language Understanding

| Benchmark | Description | Size | Year | Link |
|-----------|-------------|------|------|------|
| **DarijaMMLU** | Multiple-choice questions for general knowledge evaluation in Darija. Designed to test language models' understanding of Moroccan cultural and general knowledge. | 22,027 samples | 2024 | [Hugging Face](https://huggingface.co/datasets/MBZUAI-Paris/DarijaMMLU) |
| **DarijaHellaSwag** | Benchmark for machine reading comprehension and commonsense reasoning in Darija. Tests models' ability to predict the most plausible ending to a given context. | Not specified | 2024 | [Hugging Face](https://huggingface.co/datasets/MBZUAI-Paris/DarijaHellaSwag) |
| **DarijaBench** | Comprehensive benchmark combining test sets for translation, sentiment analysis, and summarization tasks. Provides a standardized evaluation framework across multiple NLP tasks. | Combined test sets (10% reserved) | 2024 | [Hugging Face](https://huggingface.co/datasets/MBZUAI-Paris/DarijaBench) |

### Translation Benchmarks

| Benchmark | Description | Features | Year | Link |
|-----------|-------------|----------|------|------|
| **DODa Evaluation Framework** | Evolved from the Darija Open Dataset to include standardized evaluation metrics for translation tasks. | Includes semantic and syntactic categorizations, spelling variations, verb conjugations across multiple tenses. Features both Arabic and Latin script entries. | 2024 | [arXiv](https://arxiv.org/pdf/2405.13016.pdf) |
| **TerjamaBench** | Specialized benchmark for evaluating English-Darija machine translation with focus on cultural nuance. | Supports English, Arabic-script Darija, and Latin-script Darija (Arabizi). Evaluates translation quality across diverse cultural contexts, regional variations, and linguistic phenomena. | 2025 | [Hugging Face Blog](https://huggingface.co/blog/imomayiz/terjama-bench) |

### Summarization Benchmarks

| Benchmark | Description | Features | Year | Link |
|-----------|-------------|----------|------|------|
| **GOUD.MA** | Specialized benchmark for evaluating summarization capabilities for Moroccan Darija content. | Contains over 158,000 news articles for automatic summarization in code-switched Moroccan Darija. Utilizes ROUGE (R-1, R-2, R-L) and BERTScore for systematic evaluation. | 2022 | [OpenReview](https://openreview.net/pdf?id=BMVq5MELb9) |

### LLM Evaluation Frameworks

| Framework | Description | Methodology | Year | Link |
|-----------|-------------|-------------|------|------|
| **Darija Chatbot Arena** | Evaluation platform specifically designed to benchmark LLMs' performance in understanding and generating Darija text. | Employs the Elo rating system to rank LLMs based on relative performance. Facilitates comparison of responses from various LLMs on diverse Darija prompts. Invites Moroccan communities to rate model responses to curated prompts. | 2024 | [Hugging Face Blog](https://huggingface.co/blog/atlasia/darija-chatbot-arena) / [Atlasia Blog](https://www.atlasia.ma/blog/darija-chatbot-arena) |

## 🛠️ Tools & Libraries

*Coming soon*

## 📚 Learning Resources

*Coming soon*

## 📄 Research Papers

*Coming soon*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  Made with ❤️ for the Darija NLP community
</p>


