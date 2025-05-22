# Toward Informed AV Decision-Making: Computational Model of Well-being and Trust in Mobility -- IJCAI 2025 (HAI track)

This is the official repository for paper "Toward Informed AV Decision-Making: Computational Model of Well-being and Trust in Mobility" by Zahra Zahedi, Shashank Mehrotra, Teruhisa Misu and Kumar Akash, IJCAI-25, Special Track on: Human-Centered Artificial Intelligence. <a href="https://arxiv.org/pdf/2505.14983" target="_blank">
  <img src="https://img.shields.io/badge/Paper-IJCAI25-darkcyan.svg" alt="Paper" style="max-width: 100%;">
</a>

## Description

This repository contains the data and the codes described in the paper. It is a Python implementation of dynamic Bayesian networks (DBN) and causal influence models (CIM) for inferring user wellbeing, trust, and intention in autonomous vehicle (AV) scenarios. Includes data preprocessing, model inference, statistical analysis, and visualization scripts.

## Project Structure

```
wellbeing-trust-model/
├── Paper_supplementary_files/             
│   ├── Appendix.pdf                   # Appendix of the paper (IJCAI)
│   └──  supplementary_video_IJCAI.mp4 # supplementary video of the study
├── Prolific/
│   └── data_all/                      # data collected from the study __not all data provided
├── src/                               # Source code modules
│   ├── 2TBN.py                        # DBN model implementation
│   ├── kfold_2TBN.py                  # k-fold cross-validation for DBN
│   ├── CID_2TBN.py                    # Causal Influence Model implementation
│   ├── data_preprocessing.py          # Data preprocessing
│   ├── make_data_dbn.py               # Convert processed data to DBN inputs
│   ├── stat_tests.py                  # Statistical tests functions
│   ├── main.py                        # Statistical analysis and visualization
│   ├── policy_plot.py                 # Policy decision plot generator
│   └── inferenceplot.py               # Inference result plot generator
├── fig/                               # Generated figures (PDF/PNG)
├── requirements.txt                   # Python dependencies
└── README.md                          # Project overview and instructions
```

## Installation

```bash
git clone <repo_link>
cd wellbeing-trust-model
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
tar -xzvf data.tar.gz
```

## Usage

### 1. Data Preprocessing

Clean and prepare data:
```bash
python src/data_preprocessing.py \
  --input Prolific/data_all \

```

### 2. Prepare DBN Inputs

Transform processed data for DBN modeling:
```bash
python src/make_data_dbn.py \
  --input Prolific/data_all \
```

### 3. DBN Modeling

Run cross-validation and inference:
```bash
python src/kfold_2TBN.py \
  --input Prolific/data_all \
  --folds 5
python src/2TBN.py \
  --input Prolific/data_all \
```

### 4. CIM Decision Making

Generate optimized policies:
```bash
python src/CID_2TBN.py \
  ---input Prolific/data_all \

python src/policy_plot.py \

```

### 5. Statistical Analysis & Visualization

statistical tests and ploting the inferences:
```bash
python src/main.py
python src/inferenceplot.py 
```



## Example Outputs

- `fig/owinf.pdf`, `fig/tinf.pdf`, `fig/winf.pdf`: Inference plots for others’ wellbeing, user trust, and user wellbeing.
- `fig/wpolicy.pdf`: Policy plot for wellbeing-driven decision-making.


