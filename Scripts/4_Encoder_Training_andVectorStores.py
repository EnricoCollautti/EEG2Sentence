import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import faiss
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import json
from itertools import product

# === Global Constants ===
BATCH_SIZE = 4
NUM_EPOCHS = 50
VAL_SPLIT = 0.0714
PATIENCE = 5
SEEDS = [0, 1, 2]

# === Load Data ===
def load_data():
    sentence_eeg = np.load("ZAB_SR_NR_sentence_eeg.npy")
    sentence_meta = pd.read_csv("ZAB_SR_NR_sentence_metadata.csv")
    sentence_ids = sentence_meta["sentence_id"].to_numpy()
    sentence_texts = sentence_meta["sentence_content"].astype(str).to_numpy()
    sentence_embs = np.load("ZAB_SR_NR_sentence_embeddings.npz", allow_pickle=True)["sentence_embs"]
    return sentence_eeg, sentence_embs, sentence_ids, sentence_texts
           

# === Model ===
class EEGChannelNetEncoder(nn.Module):
    def __init__(self, n_channels, embed_dim=768, input_samples=440, dropout=0.3):
        super().__init__()
        temporal_filters_per_channel = 2
        temporal_kernel_size = 5
        temporal_dilations = [1, 2, 4]
        temporal_stride = 2
        spatial_kernel_sizes = [3, 5, 7]
        spatial_filters = 8
        residual_channels = spatial_filters * len(spatial_kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.01)

        self.temp_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_channels, n_channels * temporal_filters_per_channel,
                          kernel_size=temporal_kernel_size, dilation=d, stride=temporal_stride,
                          padding=((temporal_kernel_size - 1) * d) // 2, groups=n_channels),
                nn.BatchNorm1d(n_channels * temporal_filters_per_channel), self.leaky_relu, self.dropout
            ) for d in temporal_dilations
        ])

        self.spatial_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, spatial_filters, kernel_size=(1, k), padding=(0, (k-1)//2)),
                nn.BatchNorm2d(spatial_filters), self.leaky_relu, self.dropout
            ) for k in spatial_kernel_sizes
        ])

        self.res_conv1 = nn.Sequential(
            nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(residual_channels), self.leaky_relu,
            nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(residual_channels)
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(residual_channels), self.leaky_relu,
            nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(residual_channels)
        )
        self.res_activation = self.leaky_relu

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

# === Loss ===
def cosine_loss(x, y):
    return 1 - nn.functional.cosine_similarity(
        nn.functional.normalize(x, dim=1),
        nn.functional.normalize(y, dim=1)
    ).mean()


# === Training Function ===
def train_model(config, seed, X_eeg, Y_emb, model_name="model"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Split train/val
    val_size = int(VAL_SPLIT * len(X_eeg))
    train_data, val_data = random_split(TensorDataset(X_eeg, Y_emb), [len(X_eeg) - val_size, val_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGChannelNetEncoder(X_eeg.shape[1], Y_emb.shape[1], X_eeg.shape[2], dropout=config["dropout"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config["lr_decay_factor"], patience=config["lr_patience"])
    
    best_val_loss = float("inf")
    early_stop_counter = 0
    train_loss_hist, val_loss_hist = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = cosine_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = cosine_loss(model(xb), yb)
                total_val_loss += loss.item() * xb.size(0)
        val_loss = total_val_loss / len(val_loader.dataset)

        scheduler.step(val_loss)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{model_name}_best_seed{seed}.pth")
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        print(f"[{model_name}][Seed {seed}] Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    return {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "best_val_loss": best_val_loss,
        "epochs": len(train_loss_hist)
    }

# === Plotting Function ===
def plot_loss_curves(train_loss, val_loss, title, filename):
    plt.figure()
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename)
    plt.close()

# === Generate FAISS Vector Stores ===
def generate_faiss_embeddings(model_path, model, eeg_data, batch_size, filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(eeg_data), batch_size):
            batch = torch.tensor(eeg_data[i:i+batch_size], dtype=torch.float32).to(device)
            embs = model(batch).cpu().numpy()
            embeddings.append(embs)
    embeddings = np.vstack(embeddings)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, filename)
    print(f"FAISS index saved to {filename}")
    return index, embeddings

# === Define Configurations to Try ===

# Define search space
dropouts = [0.5, 0.3]
lrs = [1e-5, 1e-6, 1e-7, 1e-4]
lr_decay_factors = [0.5, 0.1, 0.2, 0.7]
weight_decays = [1e-4, 5e-4, 5e-5]
lr_patiences = [3]

# Create all combinations
param_grid = []
for d, lr, f, wd, p in product(dropouts, lrs, lr_decay_factors, weight_decays, lr_patiences):
    param_grid.append({
        "dropout": d,
        "lr": lr,
        "lr_decay_factor": f,
        "weight_decay": wd,
        "lr_patience": p
    })


# === Load Data ===
sentence_eeg, sentence_embs, sentence_ids, sentence_texts = load_data()

# === Hold out 50 test sentences ===
all_sent_indices = np.arange(len(sentence_ids))
np.random.seed(42)  # for reproducibility
test_indices = np.random.choice(all_sent_indices, size=50, replace=False)

# Save test set
test_sentence_ids = sentence_ids[test_indices]
test_sentence_eeg = sentence_eeg[test_indices]
test_sentence_texts = sentence_texts[test_indices]
test_sentence_embs = sentence_embs[test_indices]

np.savez("heldout_test_set.npz", 
         sentence_ids=test_sentence_ids, 
         sentence_eeg=test_sentence_eeg, 
         sentence_contents=test_sentence_texts)
print(f"Held out {len(test_indices)} test sentences and saved to 'heldout_test_set.npz'.")

# Modify training data
trainval_indices = np.setdiff1d(all_sent_indices, test_indices)
sentence_eeg = sentence_eeg[trainval_indices]
sentence_embs = sentence_embs[trainval_indices]
sentence_ids = sentence_ids[trainval_indices]
sentence_texts = sentence_texts[trainval_indices]

X_sent = torch.tensor(sentence_eeg, dtype=torch.float32)
Y_sent = torch.tensor(sentence_embs, dtype=torch.float32)



# === Hyperparameter Search ===
results = []
for i, config in enumerate(param_grid):
    print(f"\nConfig {i+1}/{len(param_grid)}: {config}")
    seed_losses = []
    for seed in SEEDS:
        out = train_model(config, seed, X_sent, Y_sent, model_name=f"sent_model_config{i}")
        seed_losses.append(out["best_val_loss"])
        plot_loss_curves(out["train_loss"], out["val_loss"], 
                         title=f"Config {i} Seed {seed}", 
                         filename=f"loss_config{i}_seed{seed}.png")
    avg_loss = np.mean(seed_losses)
    std_loss = np.std(seed_losses)
    results.append((i, avg_loss, std_loss))
    print(f"Config {i} | Avg Val Loss = {avg_loss:.4f} ± {std_loss:.4f}")

# === Select Best Config ===
best_config_idx = min(results, key=lambda x: x[1])[0]
best_config = param_grid[best_config_idx]
print(f"\nBest Config: {best_config} (Index {best_config_idx})")

# === Retrain on Full Dataset using Best Config ===
avg_epochs = int(np.mean([
    train_model(best_config, seed, X_sent, Y_sent, model_name=f"retrain_check")["epochs"]
    for seed in SEEDS
]))

print(f"\nRetraining final model on full dataset for {avg_epochs} epochs...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_model = EEGChannelNetEncoder(X_sent.shape[1], Y_sent.shape[1], X_sent.shape[2], dropout=best_config["dropout"]).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])

train_loader = DataLoader(TensorDataset(X_sent, Y_sent), batch_size=BATCH_SIZE, shuffle=True)
train_loss_curve = []

final_model.train()
for epoch in range(avg_epochs):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = cosine_loss(final_model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    train_loss_curve.append(avg_loss)
    print(f"[Final] Epoch {epoch+1}/{avg_epochs} | Train Loss: {avg_loss:.4f}")

torch.save(final_model.state_dict(), "final_sent_model.pth")
plot_loss_curves(train_loss_curve, train_loss_curve, 
                 title="Final Training (Full Dataset)", 
                 filename="final_training_loss.png")

# === Generate FAISS Vector Store from Final Model ===
_, embeddings = generate_faiss_embeddings("final_sent_model.pth", final_model, sentence_eeg, BATCH_SIZE, "faiss_sentence_index_final.idx")
np.save("sentence_embeddings_final.npy", embeddings)


# === Save all results and best config for reproducibility ===
with open("results_log.json", "w") as f:
    json.dump(results, f, indent=2)

with open("best_config.json", "w") as f:
    json.dump(best_config, f, indent=2)

