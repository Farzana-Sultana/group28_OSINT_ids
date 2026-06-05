# Hybrid AI-Based Intrusion Detection System using Characteristic-Based OSINT

**Group 28 — Charles Darwin University, Faculty of Science and Technology**

A Network Intrusion Detection System (NIDS) that enriches the UNSW-NB15 dataset with
**characteristic-based Open Source Intelligence (OSINT)** — threat intelligence derived from
network *characteristics* (protocols, services, connection behaviour) rather than from host
identity (IP addresses). This makes OSINT integration possible on anonymised, privacy-preserving
benchmark datasets where conventional identity-based OSINT cannot be applied.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Contribution](#key-contribution)
3. [Repository Structure](#repository-structure)
4. [Data and Data Sources](#data-and-data-sources)
5. [The Three OSINT Features](#the-three-osint-features)
6. [Environment Configuration](#environment-configuration)
7. [Parameter Settings](#parameter-settings)
8. [Execution Steps](#execution-steps)
9. [Results Summary](#results-summary)
10. [Reproducibility](#reproducibility)
11. [Team](#team)
12. [Acknowledgements and References](#acknowledgements-and-references)

---

## Overview

Machine-learning intrusion detectors trained only on static, historical network-flow features
suffer from **temporal decay** — their accuracy drops as the threat landscape evolves. External
threat intelligence (OSINT) can supply continuously updated context, but conventional OSINT relies
on IP/domain identifiers. Benchmark datasets such as **UNSW-NB15 anonymise IP addresses** for
privacy, making identity-based OSINT impossible.

This project addresses that gap with a **characteristic-based OSINT framework**. We engineer three
OSINT features from authoritative public sources and evaluate, through a controlled two-experiment
design, whether they improve detection across four model families (Random Forest, Decision Tree,
XGBoost, and an LSTM neural network).

## Key Contribution

- The **first characteristic-based OSINT framework** for IDS on anonymised data.
- **Three independent OSINT features** from authoritative sources (NVD, FIRST.org EPSS, SANS ISC logic).
- A **reproducible, leakage-free pipeline** using a single index-based split applied identically to
  both experiments, validated across four model families.

---

## Repository Structure

```
.
├── README.md                          # This file
├── notebooks/
│   └── AI_IDS_Project.ipynb           # Main Google Colab notebook (end-to-end pipeline)
├── data/
│   ├── UNSW_NB15_training-set.csv     # Original UNSW-NB15 training partition (input)
│   └── UNSW_NB15_with_OSINT.csv       # Generated: dataset enriched with 3 OSINT features
├── results/
│   ├── model_comparison.png           # Accuracy: Experiment A vs B
│   ├── confusion_matrices.png         # Confusion matrices (4 models x 2 experiments)
│   ├── roc_curves.png                 # ROC curves with AUC scores
│   ├── shap_importance.png            # SHAP feature-importance ranking
│   ├── shap_summary.png               # SHAP summary (beeswarm) plot
│   ├── lstm_training_A.png            # LSTM training curves (baseline)
│   ├── lstm_training_B.png            # LSTM training curves (with OSINT)
│   └── model_comparison.csv           # Numeric results table
└── requirements.txt                   # Python dependencies
```

> **Note:** If the enriched dataset is large, the notebook regenerates it from the original file.
> The OSINT generation step queries public APIs and takes a few minutes.

---

## Data and Data Sources

| Source | Used for | Link |
|--------|----------|------|
| **UNSW-NB15** (training partition, 82,332 records, 45 features) | Base network-flow dataset | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| **NIST National Vulnerability Database (NVD)** | `protocol_cve_score` (CVSS severity) | https://nvd.nist.gov/developers |
| **FIRST.org EPSS API** | `service_epss_score` (exploitation probability) | https://www.first.org/epss/api |
| **SANS Internet Storm Center (logic)** | `port_behavior_risk` (behavioural weighting) | https://isc.sans.edu |

The UNSW-NB15 dataset is used under its academic research terms. All OSINT sources are public.
No personally identifiable information is used at any stage — the framework is privacy-preserving
by design.

---

## The Three OSINT Features

| Feature | Source | What it captures |
|---------|--------|------------------|
| `protocol_cve_score` | NVD / CVSS | Average vulnerability **severity** of the protocol, normalised to 0–1 |
| `service_epss_score` | FIRST.org EPSS | **Probability** the service is exploited in the near term (0–1) |
| `port_behavior_risk` | SANS ISC logic | **Behavioural** scanning/recon risk from native connection statistics |

**Port behaviour risk formula** (expert-driven weighted sum of min-max scaled inputs):

```
port_behavior_risk = 0.4 * ct_src_dport_ltm  +  0.4 * ct_dst_sport_ltm  +  0.2 * ct_state_ttl
```

The two directional connection-count signals are weighted highest (0.4 each) as direct indicators
of scanning; the connection-state TTL is a supplementary signal (0.2).

---

## Environment Configuration

The project was developed and run on **Google Colab** (Python 3.10+). It also runs locally.

### Option A — Google Colab (recommended)
1. Open `notebooks/AI_IDS_Project.ipynb` in Google Colab.
2. Colab already includes most libraries; the notebook installs any that are missing.
3. Mount Google Drive when prompted (used to cache the enriched dataset).

### Option B — Local environment
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/group28-osint-ids.git
cd group28-osint-ids

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
tensorflow>=2.12
shap>=0.42
matplotlib>=3.6
seaborn>=0.12
requests>=2.28
```

---

## Parameter Settings

### Data split
| Parameter | Value |
|-----------|-------|
| Split method | Index-based, stratified |
| Train / test ratio | 80% / 20% |
| `random_state` | 42 |
| Stratify on | binary label (Normal / Attack) |

### Feature sets
| Experiment | Features | Description |
|------------|----------|-------------|
| **A (Baseline)** | 39 | Original numeric flow features only |
| **B (Hybrid)** | 42 | 39 flow features + 3 OSINT features |

### Model hyperparameters
| Model | Key settings |
|-------|--------------|
| Random Forest | `n_estimators=100`, `random_state=42` |
| Decision Tree | default (CART), `random_state=42` |
| XGBoost | `use_label_encoder=False`, `eval_metric='logloss'`, `random_state=42` |
| LSTM | Layers: LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1, sigmoid); `epochs=50`, `batch_size=128`, optimizer `adam`, loss `binary_crossentropy` |

### Reproducibility seeds
```python
import os, random, numpy as np, tensorflow as tf
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

### OSINT API settings
| Parameter | Value |
|-----------|-------|
| NVD request delay | 6 seconds (rate-limit safe) |
| EPSS request delay | short delay between calls |
| Scaling | Min-max scaling fitted on the training partition only |

---

## Execution Steps

Run the notebook cells in order. The pipeline proceeds as follows:

1. **Load data.** Read the UNSW-NB15 training partition. A toggle (`LOAD_ORIGINAL`) controls
   whether OSINT features are regenerated or loaded from the cached enriched CSV.
2. **Clean and preprocess.** Handle data types, encode categorical fields, prepare the binary label.
3. **Generate OSINT features** (only when `LOAD_ORIGINAL = True`):
   - Query the **NVD API** per distinct protocol → `protocol_cve_score`.
   - Query the **FIRST.org EPSS API** per distinct service → `service_epss_score`.
   - Compute `port_behavior_risk` from native connection statistics using the weighted formula.
   - Save the enriched dataset to `data/UNSW_NB15_with_OSINT.csv` (cache for reproducibility).
4. **Build feature sets.** Construct Experiment A (39 features) and Experiment B (42 features).
5. **Split once.** Generate stratified train/test **indices** with `random_state=42` and apply the
   *same* indices to both A and B (`.iloc` slicing).
6. **Scale features.** Fit the scaler on the training partition only; transform train and test.
7. **Train and evaluate** Random Forest, Decision Tree, XGBoost, and LSTM under both experiments.
8. **Produce outputs.** Accuracy/precision/recall/F1, confusion matrices, ROC curves and AUC,
   SHAP analysis, and LSTM training curves — saved to `results/`.

To reproduce the published numbers exactly, set `LOAD_ORIGINAL = False` after the first run so the
cached OSINT dataset is reused (live API values can change over time).

---

## Results Summary

Test set: 16,467 records (7,400 Normal + 9,067 Attack).

| Model | Experiment A (No OSINT) | Experiment B (With OSINT) | Change |
|-------|:----------------------:|:-------------------------:|:------:|
| Random Forest | 97.58% | 97.77% | **+0.19%** |
| Decision Tree | 96.46% | 96.40% | −0.06% |
| XGBoost | 97.76% | 97.82% | **+0.06%** |
| LSTM | 95.03% | 95.62% | **+0.59%** |

- All models achieve **ROC-AUC > 0.96**; XGBoost reaches ~0.998.
- **SHAP** confirms the OSINT features are genuinely used: `port_behavior_risk` ranks **6th of 42**
  features.
- Three of four model families improve with OSINT; the small Decision Tree change is within noise.

> The improvements are modest because the baselines already exceed 95% accuracy, leaving little
> headroom. The core finding is that characteristic-based OSINT can be **integrated at all** on
> anonymised data, and that it adds genuine signal without harming performance.

---

## Reproducibility

- Fixed random seeds across Python, NumPy, and TensorFlow.
- A single index-based split shared by both experiments (only the OSINT features differ).
- The OSINT-enriched dataset is cached to disk so results do not drift with live API changes.
- MITRE ATT&CK-derived features were deliberately **removed** after they were found to cause target
  leakage (artificially perfect accuracy).

---

## Team

| Name | Student ID |
|------|------------|
| Farzana Sultana | S371957 |
| Abdul Huq Riyad | S388200 |
| Iftekhar Alam Ishti | S383561 |

**Supervisor:** Mukhtar Hussain
**Institution:** Charles Darwin University, Faculty of Science and Technology

---

## Acknowledgements and References

- Moustafa, N. & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion
  detection systems.* MilCIS, IEEE.
- Mell, P., Scarfone, K. & Romanosky, S. (2006). *Common Vulnerability Scoring System.* IEEE
  Security & Privacy.
- Jacobs, J. et al. (2021). *Exploit Prediction Scoring System (EPSS).* Digital Threats: Research
  and Practice.
- Ten, C-W., Manimaran, G. & Liu, C-C. (2010). *Cybersecurity for critical infrastructures: attack
  and defense modeling.* IEEE Transactions on Systems, Man, and Cybernetics.

Threat-intelligence data provided by the **NIST NVD**, **FIRST.org EPSS**, and the **SANS Internet
Storm Center**. UNSW-NB15 dataset provided by the **Australian Centre for Cyber Security (ACCS)**.

---

*This repository accompanies the Group 28 final report submitted for the Master of Information
Technology (Cyber Security) at Charles Darwin University.*
