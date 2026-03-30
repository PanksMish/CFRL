# CFRL-FND: Crowdsourcing and Federated Reinforcement Learning Framework for Fake News Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

This repository contains the official implementation of **CFRL-FND**, a Crowdsourcing-aided Federated Reinforcement Learning Framework for Fake News Detection. The framework integrates:

- **LLM-based Information Extraction Subsystem (IES)** — transformer-based contextual analysis with domain-aware prompting
- **Federated Reinforcement Learning (FED-RL)** — DDPG-based adaptive node selection with cost-aware optimization
- **Crowdsourcing Social Networks Pool (CSNP)** — credibility scoring using simulated annotator behavior
- **Query Data Optimization (QDO)** — standardized preprocessing pipeline
- **Score Aggregator** — weighted fusion of IES and FED-RL outputs

The proposed framework achieves **95.08% accuracy** while reducing training and communication costs by up to **20%** compared to state-of-the-art baselines.

---

## Repository Structure

```
CFRL-FND/
├── README.md
├── requirements.txt
├── config.py                          # All hyperparameters and configuration
├── dataset/
│   ├── __init__.py
│   ├── data_loader.py                 # Dataset loading and preprocessing
│   ├── federated_partitioner.py       # Non-IID data partitioning for FL
│   └── crowdsource_simulator.py       # Crowdsourcing annotator simulation
├── models/
│   ├── __init__.py
│   ├── information_extraction.py      # IES: DistilBERT-based transformer module
│   ├── ddpg_agent.py                  # Deep Deterministic Policy Gradient agent
│   ├── federated_node.py              # Virtual node for federated learning
│   └── score_aggregator.py            # Score fusion module
├── training/
│   ├── __init__.py
│   ├── federated_trainer.py           # FED-RL training loop
│   ├── node_selector.py               # Cost-aware node selection (DRLSelect)
│   └── reward_calculator.py           # Reward and cost computation
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # Accuracy, F1, Precision, Recall, ROC-AUC
│   └── cost_tracker.py                # Training & communication cost tracking
├── explainability/
│   ├── __init__.py
│   └── explainer.py                   # Attention analysis & feature attribution
├── baselines/
│   ├── __init__.py
│   ├── fed_avg.py                     # FedAvg baseline
│   ├── scaffold.py                    # SCAFFOLD baseline
│   └── centralized.py                 # Centralized baseline (CEN-FND)
├── experiments/
│   ├── run_cfrl_fnd.py                # Main experiment runner
│   ├── run_baselines.py               # Baseline comparison runner
│   └── ablation_study.py              # Ablation study runner
└── utils/
    ├── __init__.py
    ├── logger.py                      # Logging utilities
    └── seed.py                        # Reproducibility utilities
```

---

## Installation

```bash
git clone https://github.com/your-username/CFRL-FND.git
cd CFRL-FND
pip install -r requirements.txt
```

---

## Dataset

The framework uses the [Kaggle Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets), which contains:
- 21,417 real news articles
- 23,481 fake news articles
- Total: 44,898 samples

Download the dataset and place `True.csv` and `Fake.csv` inside `data/raw/`.

---

## Running Experiments

### Train CFRL-FND (proposed framework)

```bash
python experiments/run_cfrl_fnd.py --num_nodes 100 --rounds 30 --dataset_path data/raw/
```

### Run Baseline Comparisons

```bash
python experiments/run_baselines.py --dataset_path data/raw/
```

### Ablation Study

```bash
python experiments/ablation_study.py --dataset_path data/raw/
```

---

## Key Results

| Method     | Accuracy (%) | F1-Score (%) | Precision (%) | Recall (%) |
|------------|:------------:|:------------:|:-------------:|:----------:|
| CEN-FND    | 87.3         | 86.9         | 87.1          | 86.8       |
| FedAvg     | 90.4         | 90.1         | 90.2          | 90.0       |
| SCAFFOLD   | 91.2         | 90.9         | 91.0          | 90.8       |
| BRaG       | 92.5         | 92.1         | 92.3          | 92.0       |
| SheepDog   | 92.1         | 91.8         | 91.9          | 91.7       |
| **CFRL-FND** | **95.08** | **94.8**   | **94.2**      | **95.6**   |

---

## Citation

```bibtex
@inproceedings{pankaj2024cfrl,
  title={CFRL-FND: A Crowdsourcing and Federated Reinforcement Learning Framework for Fake News Detection},
  author={Pankaj et al.},
  booktitle={Proceedings of the ACM Conference},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
