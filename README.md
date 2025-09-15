# PhonemeDecoding

End-to-end pipeline for decoding **phonemes** and **acoustic features** from surface EMG.  
This repo builds on excellent prior work from:

- [dgaddy/silent_speech](https://github.com/dgaddy/silent_speech)  
- [tbenst/silent_speech](https://github.com/tbenst/silent_speech?tab=readme-ov-file)  

Much credit to them—this project would not exist without their contributions.

---

## 📂 Repository Contents

- `read_emg.py` — dataset & preprocessing (EMG/audio/phonemes)
- `data_utils.py` — feature extraction, normalizers, phoneme inventory
- `align.py` — DTW alignment utilities
- `models.py` — VQ-VAE tokenizer + EMG→phoneme model with Mixture-of-Recursions 
- `tokenizer.py` — trains and exports an EMG VQ-VAE tokenizer
- `phoneme_decoding.py` — main training loop with logging, checkpoints, metrics
- `utils.py` — tokenizer export/load + DTW loss functions

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
We provide an `environment.txt` file to reproduce the exact conda environment.

   ```bash
   conda create -n brain2speech --file environment.txt
   conda activate brain2speech


## 🚀 Training Steps

1. **Train the tokenizer**
Train a VQ-VAE tokenizer on raw EMG.

   ```bash
   python tokenizer.py
