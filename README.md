# ECG Anomaly Detection using Autoencoders

This project explores the use of Unsupervised Deep Learning to detect irregular heartbeats in ECG signals. By training an Autoencoder solely on normal heart data, the model learns to identify anomalies through reconstruction error.

## 📌 Project Overview
The goal of this project is to create a robust system that can flag potential heart issues without needing a massive labeled dataset of every possible heart condition.

### How it works:
1. **Normalization**: Scaled 140-point ECG signals using `MinMaxScaler` to help the neural network converge faster.
2. **The Autoencoder**: Built a bottleneck architecture (64 -> 32 -> 16 -> 8 neurons) that forces the model to learn the most important features of a "normal" heartbeat.
3. **Reconstruction**: The model "compresses" the input and then tries to "decompress" it back to the original shape.
4. **Anomaly Scoring**: Since the model only knows "normal," it makes large mistakes (high MAE) when it sees an "anomaly." We set a statistical threshold ($Mean + 2\sigma$) to catch these mistakes.

## 🛠️ Tools Used
* **Framework**: TensorFlow / Keras
* **Libraries**: Pandas, NumPy, Matplotlib, Scikit-Learn
* **Dataset**: ECG5000 (Time-series heartbeat data)

## 📈 Results
The model effectively separates normal heartbeats from anomalies.
* **Normal Reconstruction**: Low Error (Model is familiar with the shape).
* **Anomaly Reconstruction**: High Error (Model is unfamiliar with the shape).

## 🚀 Usage
1. Clone the repo: `git clone https://github.com/[YOUR-USERNAME]/[REPO-NAME].git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the notebook to see the plots and model performance.

---
