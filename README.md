# 🚦 Traffic Sign Recognition with Deep Learning

This project implements a Traffic Sign Recognition System using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The system is trained with deep learning (CNN in TensorFlow/Keras) and deployed via a Streamlit web app for real-time predictions.

## 📌 Features

Data Preprocessing: Handled raw GTSRB .pickle and .csv files, cleaned and balanced underrepresented classes.

Model Training:

Built a CNN baseline model from scratch.

Trained on CPU-friendly mode (resumable training supported).

Checkpointing ensures that training resumes from the last saved state.

Deployment:

Interactive Streamlit web app.

Supports image upload and webcam real-time predictions.

Displays top-3 predicted classes with probabilities.

Readable Predictions: Instead of numeric IDs (e.g., 18), the system shows actual traffic sign names (e.g., General caution).

## 📂 Dataset

The project uses the GTSRB dataset from Kaggle.
Files required:

train.pickle, valid.pickle, test.pickle

labels.pickle, label_names.csv

## ⚙️ Installation

Create a virtual environment

python -m venv tsr-env
source tsr-env/bin/activate   # Linux/Mac
tsr-env\Scripts\activate     # Windows


Install dependencies

pip install -r requirements.txt

## ▶️ Usage
1. Train the Model
python train_gtsrb_baseline.py


Training will resume automatically if interrupted.

Model checkpoints are saved in checkpoints/.

2. Run the Streamlit App
streamlit run app.py

3. Predict

Upload an image OR

Use the webcam for real-time traffic sign recognition.

📊 Results

Achieved high classification accuracy on validation/test sets.

Example Prediction:

Uploaded Image → Predicted Sign: "Speed limit 50 km/h"
Top-3 Predictions:
1. Speed limit 50 km/h (95.4%)
2. Speed limit 30 km/h (3.2%)
3. Speed limit 70 km/h (1.4%)

## 🛠️ Tech Stack

Python

TensorFlow / Keras

NumPy, Pandas, Matplotlib

Streamlit

OpenCV (for webcam input)
