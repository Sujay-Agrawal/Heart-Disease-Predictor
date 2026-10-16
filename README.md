# ❤️ Heart Disease Prediction using K-Nearest Neighbors (KNN)

A machine learning project that predicts the presence of heart disease from 13 clinical attributes, using a **K-Nearest Neighbors classifier** (91.80% test accuracy), with an interactive **Streamlit** web app for real-time predictions.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Streamlit App](#-streamlit-app)
- [Retraining the Model](#-retraining-the-model)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

Heart disease is one of the leading causes of death worldwide, and early detection matters. This project analyses patient health data (age, cholesterol, blood pressure, exercise-induced factors, etc.) and uses KNN classification to predict whether a patient is likely to have heart disease.

**Highlights**

- Exploratory data analysis with histograms, scatter plots and a correlation heatmap
- Feature standardization with `StandardScaler` (essential for distance-based models like KNN)
- Hyperparameter search over `k = 1 … 15` to find the best number of neighbors
- Trained model and scaler saved as pickle files for reuse
- Streamlit UI for entering patient details and getting an instant prediction with probability

---

## 📊 Dataset

- **File:** `heart.csv`
- **Size:** 303 rows × 14 columns
- **Missing values:** none

| Feature    | Description |
|------------|-------------|
| `age`      | Age in years |
| `sex`      | Sex (1 = male, 0 = female) |
| `cp`       | Chest pain type (4 categories) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol`     | Serum cholesterol (mg/dl) |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| `restecg`  | Resting electrocardiographic results (0–2) |
| `thalach`  | Maximum heart rate achieved |
| `exang`    | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak`  | ST depression induced by exercise relative to rest |
| `slope`    | Slope of the peak exercise ST segment (0–2) |
| `ca`       | Number of major vessels colored by fluoroscopy (0–4) |
| `thal`     | Thalassemia result (categorical) |
| `target`   | **Label:** 1 = heart disease present, 0 = absent |

---

## 🔄 Project Workflow

1. **Load data**: read `heart.csv` with pandas
2. **Explore**: check dtypes, null values, age distribution, age vs. cholesterol, feature correlations
3. **Split features/target**: `X = df.drop('target', axis=1)`, `y = df['target']`
4. **Train/test split**: 80% train / 20% test (`random_state=42`)
5. **Scale**: fit `StandardScaler` on the training set, apply to both sets
6. **Train**: `KNeighborsClassifier`
7. **Evaluate**: accuracy score on the held-out test set
8. **Tune**: loop `k` from 1 to 15 and compare accuracy
9. **Save**: export model and scaler with `pickle`
10. **Deploy**: Streamlit app loads the pickles and serves predictions

---

## 🏆 Results

The best performance was achieved with **k = 7**, giving **91.80% accuracy** on the test set.

| k | Accuracy | | k | Accuracy |
|---|----------|-|---|----------|
| 1 | 85.25% | | 9  | 90.16% |
| 2 | 81.97% | | 10 | 88.52% |
| 3 | 85.25% | | 11 | 88.52% |
| 4 | 85.25% | | 12 | 86.89% |
| 5 | 90.16% | | 13 | 88.52% |
| 6 | 90.16% | | 14 | 86.89% |
| **7** | **91.80%** | | 15 | 86.89% |
| 8 | 86.89% | | | |

> The test set is small (61 samples), so a single misclassification shifts accuracy by about 1.6 percentage points. Treat the exact figure with some caution.

---

## 📁 Project Structure

```
.
├── KNN Classifier Que (1).ipynb   # Training & evaluation notebook
├── heart.csv                      # Dataset
├── model_knn_class.pkl            # Trained KNN model
├── scaler_knn_class.pkl           # Fitted StandardScaler
├── app.py                         # Streamlit web app
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

For running the notebook you will additionally need:

```bash
pip install jupyter matplotlib seaborn numpy
```

> ⚠️ Use the **same scikit-learn version** that was used to train the model, otherwise loading the pickle files may fail or behave unexpectedly.

---

## 🚀 Usage

### Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

### Use the model in your own code

```python
import pickle
import pandas as pd

with open("model_knn_class.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler_knn_class.pkl", "rb") as f:
    scaler = pickle.load(f)

patient = pd.DataFrame([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]],
                       columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                "restecg", "thalach", "exang", "oldpeak",
                                "slope", "ca", "thal"])

prediction = model.predict(scaler.transform(patient))[0]
print("Heart disease likely" if prediction == 1 else "Heart disease unlikely")
```

---

## 🖥️ Streamlit App

The app provides:

- A three-column form for demographics, vitals/blood tests and exercise test results
- Dropdowns and sliders instead of raw numeric codes
- A clear result card (risk likely / unlikely)
- Probability of heart disease with a progress bar
- An expandable view of the exact values sent to the model

Inputs are passed to the model **in the same column order used during training**, then scaled with the saved scaler.

---

## 🔁 Retraining the Model

Run all cells in `KNN Classifier Que (1).ipynb`. When saving, make sure you pickle the model trained with your chosen `k`:

```python
import pickle

# Re-train with the best k found during tuning
best_model = KNeighborsClassifier(n_neighbors=7)
best_model.fit(X_train, y_train)

with open("model_knn_class.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler_knn_class.pkl", "wb") as f:
    pickle.dump(scalar, f)
```

---

## ⚠️ Limitations

- **Small dataset** (303 patients), so results may not generalise to other populations
- **Single train/test split**: cross-validation would give a more reliable estimate
- Accuracy alone is reported; for a medical use case, **recall/sensitivity** (not missing sick patients) matters at least as much
- KNN stores the full training set and gets slower with large datasets
- Categorical features (`cp`, `restecg`, `slope`, `thal`) are used as numeric codes, which implies an ordering that may not exist

---

## 🔮 Future Improvements

- Cross-validation and `GridSearchCV` for `k`, distance metric and weighting
- Report precision, recall, F1-score and ROC-AUC with a confusion matrix
- One-hot encode categorical features
- Compare against Logistic Regression, Random Forest and Gradient Boosting
- Wrap scaler + model in a single scikit-learn `Pipeline`
- Add model explainability (e.g., permutation importance, SHAP)
- Deploy the app on Streamlit Community Cloud

---

## 🩺 Disclaimer

This project is for **educational purposes only**. It is **not** a medical device and must not be used for diagnosis or treatment decisions. Always consult a qualified healthcare professional.

---

## 🛠️ Built With

- [Python](https://www.python.org/)
- [pandas](https://pandas.pydata.org/) · [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/) · [Seaborn](https://seaborn.pydata.org/)
- [Streamlit](https://streamlit.io/)
