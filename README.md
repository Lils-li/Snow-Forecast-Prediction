# Snow-Forecast-ML

Predict snowfall using a neural network with Z-score normalization, train/validation/test split, and TensorFlow model quantization.

---

## Project Structure

Snow-Forecast-ML/  
├─ data/  
│ ├─ dataset.csv  
│ └─ weather_dataset_UNBALANCED_scaled.csv  
├─ notebooks/  
│ └─ tflite_lab6_tasks.ipynb  
├─ README.md  

- `data/` — contains the CSV datasets used for training and testing.  
- `notebooks/` — Jupyter notebook with all preprocessing, model training, evaluation, and TensorFlow Lite conversion steps.

---

## Features

- Z-score normalization of input features.  
- Train/Validation/Test split for model evaluation.  
- Binary classification model to predict snowfall (`Yes`/`No`).  
- TensorFlow neural network with one hidden layer and dropout for regularization.  
- Model quantization to INT8 for deployment on microcontrollers.

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
# or manually:
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

2. Open the notebook:

jupyter notebook notebooks/tflite_lab6_tasks.ipynb

3. Follow the steps in the notebook:
Load datasets from data/.
Normalize features using Z-score.
Train the neural network.
Evaluate accuracy, loss, and confusion matrix.
Convert model to TensorFlow Lite (optional quantization).

Notes
.tflite and .keras files are not included — they can be generated from the notebook.
Graphs like accuracy/loss plots are generated dynamically in the notebook.
The notebook is self-contained: all preprocessing, training, evaluation, and export steps are included.
