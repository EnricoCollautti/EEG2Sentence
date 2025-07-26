import mat73
import numpy as np
import pandas as pd

# ==== CONFIG ====
file_paths = [
    "/work3/s247362/SubjectZAB/resultsZAB_SR.mat",
    "/work3/s247362/SubjectZAB/resultsZAB_NR.mat"
]

# === OUTPUT ===
sentence_proc_list = []
sentence_texts = []
sentence_ids = []

word_proc_list = []
word_metadata = []

global_sentence_id = 1
global_word_id = 1

# === PROCESSING ===
for mat_file in file_paths:
    print(f"\nProcessing: {mat_file}")
    data = mat73.loadmat(mat_file)
    sentence_data = data["sentenceData"]
    num_sentences = len(sentence_data["content"])

    for s_idx in range(num_sentences):
        # Extract ICA components and weights
        try:
            ic_act = np.array(sentence_data["IC_act_automagic"][s_idx])
            ica_weights = np.array(sentence_data["icaweights"][s_idx])
            if ic_act is None or ica_weights is None or ic_act.size == 0 or ica_weights.size == 0:
                continue
        except:
            continue

        # Reconstruct sentence-level EEG: ICA unmixing inverse * component activations
        try:
            ica_winv = np.linalg.pinv(ica_weights)
            reconstructed_sentence = ica_winv @ ic_act
        except:
            continue

        # Save sentence-level preprocessed data
        sentence_proc_list.append(reconstructed_sentence)
        sentence_texts.append(sentence_data["content"][s_idx])
        sentence_ids.append(global_sentence_id)
        current_sentence_id = global_sentence_id
        global_sentence_id += 1

        # Extract word structure
        words_struct = sentence_data["word"][s_idx]
        num_words = len(words_struct["content"])

        for w_idx in range(num_words):
            word_text = words_struct["content"][w_idx]
            raw_ic_list = words_struct.get("IC_act_automagic", [])[w_idx]

            if raw_ic_list is None or not isinstance(raw_ic_list, list):
                continue

            for fix_idx, ic_fix in enumerate(raw_ic_list):
                if ic_fix is None:
                    continue
                ic_fix_array = np.array(ic_fix)
                if ic_fix_array.size == 0 or np.isnan(ic_fix_array).all():
                    continue

                # Reconstruct word-level fixation EEG using sentence-level ica_winv
                try:
                    proc_fix_array = ica_winv @ ic_fix_array
                except:
                    continue

                word_proc_list.append(proc_fix_array)
                word_metadata.append((global_word_id, word_text, current_sentence_id, fix_idx + 1))
                global_word_id += 1

# === PADDING ===

# Sentence padding
max_T_sent = max(eeg.shape[1] for eeg in sentence_proc_list)
sentence_array = np.stack([
    np.pad(eeg, ((0, 0), ((max_T_sent - eeg.shape[1]) // 2,
                          (max_T_sent - eeg.shape[1]) - (max_T_sent - eeg.shape[1]) // 2)),
           mode='constant')
    for eeg in sentence_proc_list
])

# Word padding
max_T_word = max(eeg.shape[1] for eeg in word_proc_list)
word_array = np.stack([
    np.pad(eeg, ((0, 0), ((max_T_word - eeg.shape[1]) // 2,
                          (max_T_word - eeg.shape[1]) - (max_T_word - eeg.shape[1]) // 2)),
           mode='constant')
    for eeg in word_proc_list
])

# Metadata
df_sentence = pd.DataFrame({"sentence_id": sentence_ids, "sentence_content": sentence_texts})
df_word = pd.DataFrame(word_metadata, columns=["word_id", "word_content", "sentence_id", "fixation_idx"])

# === SAVE ===
np.save("ZAB_SR_NR_sentence_eeg.npy", sentence_array)
np.save("ZAB_SR_NR_word_eeg.npy", word_array)
df_sentence.to_csv("ZAB_SR_NR_sentence_metadata.csv", index=False)
df_word.to_csv("ZAB_SR_NR_word_metadata.csv", index=False)

print("EEG and metadata saved.")

