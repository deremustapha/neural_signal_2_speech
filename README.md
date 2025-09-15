# neural_signal_2_speech


# PhonemeDecoding

End-to-end pipeline for decoding **phonemes** and **acoustic features** from surface EMG.  
This repo builds on excellent prior work from:

- [dgaddy/silent_speech](https://github.com/dgaddy/silent_speech)  
- [tbenst/silent_speech](https://github.com/tbenst/silent_speech?tab=readme-ov-file)  

Much credit to them—this project would not exist without their contributions.

---

## 📂 Repository Contents

- `read_emg.py` — dataset & preprocessing (EMG/audio/phonemes):contentReference[oaicite:0]{index=0}  
- `data_utils.py` — feature extraction, normalizers, phoneme inventory:contentReference[oaicite:1]{index=1}  
- `align.py` — DTW alignment utilities:contentReference[oaicite:2]{index=2}  
- `models.py` — VQ-VAE tokenizer + EMG→phoneme model with Mixture-of-Recursions:contentReference[oaicite:3]{index=3}  
- `tokenizer.py` — trains and exports an EMG VQ-VAE tokenizer:contentReference[oaicite:4]{index=4}  
- `phoneme_decoding.py` — main training loop with logging, checkpoints, metrics:contentReference[oaicite:5]{index=5}  
- `utils.py` — tokenizer export/load + DTW loss functions:contentReference[oaicite:6]{index=6}  

---

## 📥 Data Setup

1. **Download Digital Voicing dataset**  
   From [Zenodo](https://zenodo.org/records/4064409).


   
2. **Get text alignments + files from `dgaddy/silent_speech`**  
Download:
- ` TextGrid alignments`  
- `testset_largedev.json`  
- `normalizers.pkl`  



3. **Set paths in code**  
- In `read_emg.py`, configure  (`--silent_data_directories`, `--voiced_data_directories`, `--testset_file`, `--text_align_directory`)
- In `data_utils.py`, set `--normalizers_file` path to `normalizers.pkl`

---

## ⚙️ Installation

1. **Create Conda environment**

```bash
conda create -n phoneme-decoding python=3.10 -y
conda activate brain2speech







