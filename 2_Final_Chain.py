import os
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download


# ---------------------- CONFIG ----------------------
SENTENCE_MODEL_PATH = "official_final_model.pth"
TEST_SET_PATH = "OFFICIAL_heldout_test_set.npz"
SENTENCE_EEG_PATH = "ZAB_SR_NR_sentence_eeg.npy"
SENTENCE_META_PATH = "ZAB_SR_NR_sentence_metadata.csv"
SENTENCE_INDEX_PATH = "faiss_sentence_index_official.idx"
OUTPUT_CSV = "final_test_results_sentonly.csv"
SIMILAR_SENTENCES_CSV = "top3_similar_sentences.csv"

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_DIR = "./mixtral_local"  # local storage directory
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token_here" 

# Download the full model if not already present
if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
    print(f"Downloading {MODEL_ID} to {MODEL_DIR} ...")
    
    snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,  
    resume_download=True,          
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)



# ---------------------- LLM LOADING ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.2,
    device_map="auto"
)

# ------------------ EEG Encoder ------------------
class EEGChannelNetEncoder(nn.Module):
    def __init__(self, n_channels, embed_dim=768, input_samples=440):
        super().__init__()
        temporal_filters_per_channel = 2
        temporal_kernel_size = 5
        temporal_dilations = [1, 2, 4]
        temporal_stride = 2
        spatial_kernel_sizes = [3, 5, 7]
        spatial_filters = 8
        residual_channels = spatial_filters * len(spatial_kernel_sizes)

        self.temp_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_channels, n_channels * temporal_filters_per_channel,
                          kernel_size=temporal_kernel_size, dilation=d, stride=temporal_stride,
                          padding=((temporal_kernel_size - 1) * d) // 2, groups=n_channels),
                nn.BatchNorm1d(n_channels * temporal_filters_per_channel), nn.ReLU()
            ) for d in temporal_dilations
        ])

        self.spatial_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, spatial_filters, kernel_size=(1, k), padding=(0, (k - 1) // 2)),
                nn.BatchNorm2d(spatial_filters), nn.ReLU()
            ) for k in spatial_kernel_sizes
        ])

        self.res_conv1 = nn.Sequential(
            nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(residual_channels), nn.ReLU(),
            nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(residual_channels)
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(residual_channels), nn.ReLU(),
            nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(residual_channels)
        )
        self.res_activation = nn.ReLU()

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, input_samples)
            tmp = torch.cat([layer(dummy) for layer in self.temp_convs], dim=1).permute(0, 2, 1).unsqueeze(1)
            tmp = torch.cat([layer(tmp) for layer in self.spatial_convs], dim=1)
            r1 = self.res_activation(self.res_conv1(tmp) + tmp)
            r2 = self.res_activation(self.res_conv2(r1) + r1)
            linear_in = r2.mean(dim=3).view(1, -1).size(1)
        self.fc = nn.Linear(linear_in, embed_dim)

    def forward(self, x):
        x = torch.cat([layer(x) for layer in self.temp_convs], dim=1).permute(0, 2, 1).unsqueeze(1)
        x = torch.cat([layer(x) for layer in self.spatial_convs], dim=1)
        r1 = self.res_activation(self.res_conv1(x) + x)
        r2 = self.res_activation(self.res_conv2(r1) + r1)
        return self.fc(r2.mean(dim=3).view(x.size(0), -1))

# ------------------ LLM Refinement ------------------
def refine_sentence_from_top3(similar_sentences: list[str]):
    context = "\n".join(f"{i+1}. {s}" for i, s in enumerate(similar_sentences))
    prompt = (
        "You are given three similar sentences.\n"
        "Your task is to generate one single fluent, grammatically correct sentence "
        "that captures the same meaning, without adding or removing any information. "
        "Only output the corrected sentence.\n\n"
        f"Similar sentences:\n{context}\n\n"
        "Final refined sentence:"
    )
    result = llama_pipeline(prompt, return_full_text=False)
    return result[0]["generated_text"].strip().strip('"')

# ------------------ Main Inference ------------------
def run_test():
    print("Loading models and data...")
    test_data = np.load(TEST_SET_PATH, allow_pickle=True)
    test_ids = test_data["sentence_ids"]
    test_texts = test_data["sentence_contents"]
    test_eeg_sent = test_data["sentence_eeg"]

    faiss_sent = faiss.read_index(SENTENCE_INDEX_PATH)
    sentence_candidates = pd.read_csv(SENTENCE_META_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sent_model = EEGChannelNetEncoder(test_eeg_sent.shape[1], input_samples=test_eeg_sent.shape[2]).to(device)
    sent_model.load_state_dict(torch.load(SENTENCE_MODEL_PATH, map_location=device))
    sent_model.eval()

    results_main = []
    results_top3 = []

    print("Generating results (sentence-level only)...")
    for i in tqdm(range(len(test_ids))):
        sid = test_ids[i]
        original_text = test_texts[i]
        eeg_sent = torch.tensor(test_eeg_sent[i:i+1], dtype=torch.float32).to(device)
        eeg_sent_emb = sent_model(eeg_sent).cpu().detach().numpy()

        _, sent_nn = faiss_sent.search(eeg_sent_emb, 4)
        filtered_indices = [
            idx for idx in sent_nn[0]
            if sentence_candidates.iloc[idx]['sentence_content'].strip() != original_text.strip()
        ][:3]
        similar_sentences = sentence_candidates.iloc[filtered_indices]['sentence_content'].tolist()

        refined = refine_sentence_from_top3(similar_sentences)

        results_main.append({
            "sentence_id": int(sid),
            "original_sentence": original_text,
            "refined_from_top3": refined
        })

        top3_record = {"sentence_id": int(sid), "original_sentence": original_text}
        for j in range(len(similar_sentences)):
            top3_record[f"similar_{j+1}"] = similar_sentences[j]
        results_top3.append(top3_record)

    pd.DataFrame(results_main).to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(results_top3).to_csv(SIMILAR_SENTENCES_CSV, index=False)
    print(f"Saved:\n- Refined sentences → {OUTPUT_CSV}\n- Top 3 similar sentences → {SIMILAR_SENTENCES_CSV}")


if __name__ == "__main__":
    run_test()

