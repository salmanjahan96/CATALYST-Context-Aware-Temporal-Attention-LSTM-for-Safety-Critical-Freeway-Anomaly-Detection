# CATALYST: Context-Aware Temporal Attention LSTM for Safety‑Critical Freeway Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**CATALYST** is a deep learning framework devised to spot anomalies (e.g., traffic crashes) on freeways using real‑time loop detector data. The model merges **Bidirectional LSTM** with a **Bahdanau attention mechanism** to grasp spatio‑temporal dependencies and concentrate on vital time steps. The pipeline tackles severe class imbalance (1000:1) via **Random Under‑Sampling + SMOTE** and attains state‑of‑the‑art outcomes on the Nashville freeway anomaly dataset.

---

## 🚀 Key Features

- **Hybrid Architecture** – BiLSTM + Bahdanau Attention + LSTM for sturdy sequence modeling.
- **Extreme Imbalance Handling** – RUS + SMOTE strategy to avoid preference toward majority class.
- **Advanced Feature Engineering** – Rolling statistics, speed–volume ratios, peak hour indicators, etc.
- **Threshold Optimization** – F1‑score maximization for the optimal decision boundary.
- **Reproducible Pipeline** – Modular code, configurable parameters, and full evaluation suite.

---

## 📊 Performance Summary

| Model      | Accuracy | F1‑Score | Recall | Precision | AUC‑ROC |
|------------|----------|----------|--------|-----------|---------|
| Baseline   | 0.9436   | 0.9466   | 1.0000 | 0.8986    | 0.9796  |
| **Enhanced (Ours)** | **0.9530** | **0.9551** | **1.0000** | **0.9140** | 0.9764 |

> The enhanced model detects **100% of anomalies** (recall = 1.0) while sustaining high precision.

---
