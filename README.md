# 🧠 EEG-to-Text Decoding Pipeline

This repository contains a complete pipeline for decoding EEG signals into natural language sentences using deep learning and NLP techniques. The system is designed for offline simulation of a Brain-Computer Interface (BCI), enabling the generation of textual output based on non-invasive brain activity recorded during reading tasks.

---

## 📦 Repository Contents

| Script | Description |
|--------|-------------|
| `1_Database_Creation.py` | Preprocesses raw `.mat` EEG files and generates padded EEG matrices and metadata at both the sentence and word level. |
| `2_Final_Chain.py` | Runs the final evaluation pipeline: loads a trained encoder, embeds EEG, retrieves top-3 similar sentences using FAISS, and uses a local LLM (Mixtral) to generate a refined sentence. |
| `3_NLP_Embeddings.py` | Generates contextual embeddings for all sentence and word labels using the `all-mpnet-base-v2` transformer model. |
| `4_Encoder_Training_andVectorStores.py` | Trains the EEG encoder model using cosine similarity loss and saves a FAISS index for efficient retrieval of sentence embeddings. |

---

## 🔄 Pipeline Overview

### 🧪 Step 1: Data Creation

```bash
python 1_Database_Creation.py
```

- Loads `.mat` EEG files processed with ICA
- Reconstructs sentence- and word-level EEG signals
- Applies padding to ensure uniform shape
- Saves:
  - `ZAB_SR_NR_sentence_eeg.npy`
  - `ZAB_SR_NR_word_eeg.npy`
  - `ZAB_SR_NR_sentence_metadata.csv`
  - `ZAB_SR_NR_word_metadata.csv`

---

### 🧠 Step 2: NLP Embeddings

```bash
python 3_NLP_Embeddings.py
```

- Loads preprocessed sentence and word metadata
- Uses `sentence-transformers/all-mpnet-base-v2` to generate:
  - Sentence embeddings
  - Word embeddings (aligned with sentence context)
- Saves:
  - `ZAB_SR_NR_sentence_embeddings.npz`
  - `ZAB_SR_NR_word_embeddings.npz`

---

### 📈 Step 3: Encoder Training & Indexing

```bash
python 4_Encoder_Training_andVectorStores.py
```

- Trains an EEG encoder (`EEGChannelNetEncoder`) to map EEG signals to their matching sentence embeddings
- Uses cosine similarity as the loss function
- Saves:
  - `official_final_model.pth` (best model)
  - `sentence_embeddings_official.npy`
  - `faiss_sentence_index_official.idx`
  - `official_final_train_val_loss.png`
  - `official_final_config.json`

---

### 🤖 Step 4: Sentence Decoding (LLM)

```bash
python 2_Final_Chain.py
```

- Loads EEG test data and trained encoder
- Embeds EEG, retrieves top-3 semantically similar sentences
- Prompts a **local Mixtral-8x7B-Instruct** LLM to output a fluent refined sentence
- Saves:
  - `final_test_results_sentonly.csv`
  - `top3_similar_sentences.csv`

---

## 🧩 Dependencies

Install required libraries with:

```bash
pip install -r requirements.txt
```

### Suggested `requirements.txt`:

```txt
numpy
pandas
torch
sentence-transformers
transformers
faiss-cpu
matplotlib
mat73
tqdm
```

---

## 🖥️ Local Model Note

This pipeline uses a **locally downloaded** version of [`mistralai/Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1). Download it manually or via:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    local_dir="./mixtral_local",
    local_dir_use_symlinks=False,
    resume_download=True,
    token="your_HF_token"
)
```

Make sure to set the `HUGGINGFACEHUB_API_TOKEN` environment variable in your script or shell.

---

## 🧪 Test Set

A held-out test set is expected at:

```
OFFICIAL_heldout_test_set.npz
```

It must include:
- `sentence_ids`
- `sentence_contents`
- `sentence_eeg`

---

## 📌 Author

**Enrico Collautti**  
Biomedical Engineering & AI  
Thesis Project (2025)  
University of Padova · DTU · Boston University

---

## 📜 License

This repository is for academic and research purposes only.

---

## 🔗 Citation

If you use this code or adapt the methodology, please cite the original project or contact the author.
