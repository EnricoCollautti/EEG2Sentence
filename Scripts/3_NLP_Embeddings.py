import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# === Load Sentence-Level Data ===
sentence_metadata = pd.read_csv("ZAB_SR_NR_sentence_metadata.csv")
sentence_eeg = np.load("ZAB_SR_NR_sentence_eeg.npy")  

# === Load Word-Level Data ===
word_metadata = pd.read_csv("ZAB_SR_NR_word_metadata.csv")
word_eeg = np.load("ZAB_SR_NR_word_eeg.npy")  

# === Load Model and Tokenizer ===
model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModel.from_pretrained(model_name)

# === Sentence Embeddings ===
sentence_texts = sentence_metadata["sentence_content"].tolist()
sentence_embeddings = model.encode(sentence_texts, show_progress_bar=True)

# Save sentence embeddings
np.savez("ZAB_SR_NR_sentence_embeddings.npz",
         sentence_embs=sentence_embeddings,
         sentence_ids=sentence_metadata["sentence_id"].to_numpy(),
         sentence_contents=sentence_metadata["sentence_content"].to_numpy())

# === Contextual Word Embeddings ===
contextual_word_embeddings = []
word_ids = []
word_contents = []

for i, row in tqdm(word_metadata.iterrows(), total=len(word_metadata)):
    word = row["word_content"]
    sent_id = row["sentence_id"]
    word_id = row["word_id"]

    # Match sentence
    sent_row = sentence_metadata[sentence_metadata["sentence_id"] == sent_id]
    if sent_row.empty:
        contextual_word_embeddings.append(np.zeros(768))
        word_ids.append(word_id)
        word_contents.append(word)
        continue

    sentence = sent_row["sentence_content"].values[0]
    words = sentence.split()

    # Tokenize with alignment
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    word_ids_map = encoded.word_ids()

    with torch.no_grad():
        output = hf_model(**encoded)

    token_embeddings = output.last_hidden_state.squeeze(0)  

    # Find tokens for this word
    try:
        word_index_in_sent = words.index(word)
        token_idxs = [i for i, idx in enumerate(word_ids_map) if idx == word_index_in_sent]
        if token_idxs:
            word_embed = token_embeddings[token_idxs].mean(dim=0)
        else:
            word_embed = torch.zeros(768)
    except ValueError:
        word_embed = torch.zeros(768)

    contextual_word_embeddings.append(word_embed.numpy())
    word_ids.append(word_id)
    word_contents.append(word)

# Save word embeddings
contextual_word_embeddings = np.stack(contextual_word_embeddings)
np.savez("ZAB_SR_NR_word_embeddings.npz",
         word_embs=contextual_word_embeddings,
         word_ids=np.array(word_ids),
         word_contents=np.array(word_contents))

