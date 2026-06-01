# ML-Based Heart Disease Prediction

A machine learning project for **early heart disease risk assessment** using clinical patient data from a multispecialty hospital in India. The workflow covers exploratory analysis, model training, and evaluation in a Jupyter notebook.

## Project Overview

This repository implements an ML-based clinical decision support prototype that predicts whether a patient is likely to have heart disease based on features such as age, gender, blood pressure, cholesterol, resting ECG, maximum heart rate, and related clinical indicators.

The dataset contains approximately **1,000 patient records** with labeled outcomes, making it suitable for supervised learning and healthcare analytics demonstrations.

## Features

- End-to-end Jupyter workflow (`heartdisease.ipynb`)
- Structured cardiovascular dataset (`Cardiovascular_Disease_Dataset.csv`)
- Data preprocessing and train/test split
- Supervised classification for binary heart disease detection
- Model evaluation with standard scikit-learn metrics

## Project Structure

```
ML-Based-Heart-Disease-Prediction/
├── Cardiovascular_Disease_Dataset.csv   # Clinical dataset
├── heartdisease.ipynb                   # Analysis, training, and evaluation
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/aditya-soni11/ML-Based-Heart-Disease-Prediction.git
   cd ML-Based-Heart-Disease-Prediction
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

4. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `heartdisease.ipynb` and run all cells.

## Machine Learning Model

The notebook trains a **Logistic Regression** classifier (`sklearn.linear_model.LogisticRegression`) after splitting the data with `train_test_split`.

**Pipeline summary:**

1. Load and inspect `Cardiovascular_Disease_Dataset.csv`
2. Prepare feature matrix `X` and target `y`
3. Split into training and test sets (80/20)
4. Train Logistic Regression (`max_iter=1000`)
5. Evaluate using accuracy and classification metrics

Logistic Regression is used because it is interpretable, efficient on tabular clinical data, and well suited for binary medical risk classification. The notebook also explores **Random Forest** for comparison.

> **Note:** This project is for educational and research purposes only. It is not a substitute for professional medical diagnosis.

## Results

Model performance is reported in the notebook (approximately **96% accuracy** on the test split). Re-run the notebook to reproduce metrics in your environment.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with a clear description of your changes.

## License

Add an appropriate open-source license before public distribution if not already present.
