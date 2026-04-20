# 🎬 MovieInsightML — Revenue Forecasting & Genre Classification with TinyML Deployment

---

## 📌 Project Overview

MovieInsightML is an end-to-end machine learning pipeline that combines **text and numeric features** from a movie metadata dataset to perform:

- 📈 **Regression** — Predict a movie's revenue using budget, popularity, runtime, vote stats, and NLP-extracted text features
- 🏷️ **Classification** — Predict the main genre of a movie (20-class problem)
- 📦 **TinyML Deployment** — Convert the best regression model to TensorFlow Lite (`.tflite`) and evaluate it on simulated edge devices via Edge Impulse

---

## 📁 Repository Structure

```
MovieInsightML/
│
├── ml copy.ipynb                        # Regression pipeline (full workflow)
├── clss.ipynb                           # Classification pipeline
│
├── linear_regression_float32.tflite     # TFLite model (float32)
├── linear_regression_int8.tflite        # TFLite model (int8 quantized)
│
├── ML_PROJECT.pdf                       # Full project report
└── README.md
```

---

## 🧠 Models Used

### Regression (Revenue Forecasting)
| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Linear Regression | 0.7022 | 12,322,909 | 1,389,020 |
| Histogram Decision Tree | 0.6831 | 12,699,825 | 689,847 |
| Random Forest | 0.6843 | 12,676,467 | 679,583 |
| Neural Network | 0.7003 (val: 0.6761) | — | — |

### Classification (Genre Prediction — 20 classes)
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 0.3278 | 0.0665 |
| K-Nearest Neighbors | 0.3239 | 0.1160 |
| Decision Tree | 0.3414 | 0.0918 |

> Classification accuracy reflects the difficulty of 20-class genre prediction from text and numeric features alone.

---

## 🔬 Feature Engineering

**From overview text:**
- Word count
- Average word length
- Sentiment score (TextBlob polarity)
- TF-IDF vectorization + SVD dimensionality reduction

**From numeric fields:**
- Budget, popularity, runtime, vote_average, vote_count, release_year

**Target variables:**
- `revenue` (continuous) — regression
- `main_genre` (20 classes) — classification

---

## 📦 TinyML Deployment

The Linear Regression model was converted to TensorFlow Lite and evaluated on Edge Impulse:

| Runtime | Compiler | RAM | ROM/Flash | Latency |
|---------|----------|-----|-----------|---------|
| Low-end MCU | EON Compiler | Lower | Smaller | Competitive |
| Low-end MCU | TFLite | +1.3–1.4K more | +11–15K more | Similar |
| AI Accelerator | EON | Minimal | 1.4K | 1 ms |

**Key finding:** EON Compiler consistently outperforms standard TFLite on memory-constrained microcontrollers, making it preferable for real edge deployments.

---

## ⚙️ Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow textblob nltk
pip install scipy joblib
```

---

## ▶️ How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/MovieInsightML.git
cd MovieInsightML
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run Regression Notebook**
```bash
jupyter notebook "ml copy.ipynb"
```

4. **Run Classification Notebook**
```bash
jupyter notebook clss.ipynb
```

5. **TFLite models** are ready to use directly — upload `.tflite` files to [Edge Impulse](https://edgeimpulse.com) for edge evaluation.

---

## 📊 Key Results

- ✅ Linear Regression achieved **R² = 0.70** on revenue prediction with 5-fold CV stability (std = 0.015)
- ✅ Decision Tree achieved highest classification accuracy at **34.14%** across 20 genres
- ✅ Model successfully converted to TFLite (float32 + int8 quantized) with significant size reduction
- ✅ EON Compiler deployment showed superior memory efficiency on simulated microcontrollers

---

## 📄 Report

Full methodology, results, and analysis are documented in [`ML_PROJECT.pdf`](./ML_PROJECT.pdf).

---

> *Submitted for ML Lab — Fall 2025 | FAST-NUCES Islamabad*
