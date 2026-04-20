Movie Revenue Prediction & Genre Classification Using Machine Learning and TinyML
Short Description

This project develops a complete machine learning pipeline for:

1. Movie Revenue Prediction (Regression Task)

Using numeric metadata and text-derived features from movie overviews, several models were trained without using external ML libraries such as scikit-learn. The models include:

Linear Regression (scratch implementation)

Histogram Decision Tree Regressor (scratch)

Random Forest Regressor (scratch)

TinyML-compatible TensorFlow Lite regression model

After optimization, the decision-tree-based model was selected for TinyML deployment due to its balance of accuracy and computational efficiency. The final model was converted into TensorFlow Lite (Float32, Dynamic Range, and Int8) formats for microcontroller deployment.

2. Movie Genre Classification (Classification Task)

A separate classification pipeline was implemented to predict a movie’s primary genre from numeric and NLP features. Models include:

Softmax Logistic Regression (scratch, SVD-reduced features)

KNN Classifier (scratch)

Decision Tree Classifier (scratch)

Random Forest Classifier (scratch)

The classifier predicts one of 20 possible genres.

3. Data Preprocessing Pipeline

A robust preprocessing pipeline was implemented to ensure:

No data leakage between train and test sets

Memory-efficient operations to avoid kernel crashes

Cleaned and normalized numeric features

TF-IDF extraction from overview text (train-only vocabulary)

Dimensionality reduction using SVD for TinyML deployment

Proper label encoding for genre classification

Required Libraries and Dependencies
Core Libraries
numpy
pandas
re
collections
gc

Text Processing
textblob

Sparse Matrix and Linear Algebra
scipy

TinyML / Model Conversion
tensorflow
tensorflow-lite

Optional (for debugging and visualization)
matplotlib
tqdm

Installation
pip install numpy pandas textblob scipy tensorflow matplotlib tqdm

Instructions to Run the Code
1. Place Dataset Files

Ensure the project directory contains:

train.csv
test.csv

2. Run Preprocessing

This step performs all data cleaning, text feature extraction, TF-IDF vectorization, normalization, and SVD dimensionality reduction.

To run preprocessing:

python preprocessing.py


If using Jupyter Notebook, simply execute the preprocessing cells.

Preprocessing outputs include:

X_train_np.pkl
y_train_np.pkl
X_train_clf.pkl
y_genre_train.pkl

3. Train Regression Models

This trains:

Linear Regression

Histogram Decision Tree

Random Forest

Run:

python train_regressors.py


Metrics reported:

R² score

MAE

RMSE

Training time

Cross-validation results

4. Train Classification Models

This trains:

Softmax Logistic Regression (SVD features)

KNN Classifier

Decision Tree Classifier

Random Forest Classifier

Run:

python train_classifiers.py


The script prints:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

5. Export Models for TinyML

To generate TensorFlow Lite versions of the trained regression model:

python export_tinyml.py


Generated files:

tflite_models/model_float32.tflite
tflite_models/model_dynamic_range.tflite
tflite_models/model_int8.tflite


These files are ready for deployment on microcontrollers.

6. Deploy to Hardware (TinyML)

Use Arduino IDE or Edge Impulse.

Example:

xxd -i model_int8.tflite > model_int8.h


Then include in your microcontroller code:

#include "model_int8.h"


Run inference by feeding a 209-dimensional normalized feature vector to the model.

Suggested Project Structure
/movie-ml-project
│── preprocessing.py
│── train_regressors.py
│── train_classifiers.py
│── export_tinyml.py
│── utils.py
│── train.csv
│── test.csv
│── tflite_models/
│   ├── model_float32.tflite
│   ├── model_dynamic_range.tflite
│   ├── model_int8.tflite
│── README.md