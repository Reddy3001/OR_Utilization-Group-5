# OptiOR

**A machine learning initiative to optimize Operating Room (OR) utilization by predicting actual surgical case durations.**

This repository contains the complete analysis, statistical testing, and machine learning workflow used to improve scheduling accuracy. It compares historical booked times against actual durations and deploys predictive models to reduce idle time and delays.

---

## 🏥 Project Purpose

This project supports the **A3 countermeasure "Integrated Predictive Scheduling System."**
By moving beyond static averages to predictive modeling, we aim to deliver:
* **Accurate Predictions:** minimizing the gap between booked and actual surgery times.
* **Optimized Scheduling:** Improved OR utilization forecasting.
* **Operational Efficiency:** Reduced day-of-surgery delays and idle time gaps.

---

## 📁 Project Structure

```text
.
├── data/
│   └── 2022_Q1_OR_Utilization.csv       # Raw dataset
│
├── notebooks/
│   └── OR_Utilisation_Analysis.ipynb    # Main analysis & modeling workflow
│
├── scripts/
│   └── model_pipeline.py                # (Optional) Standalone pipeline script
│
├── models/
│   └── best_or_time_prediction_model.pkl # Trained model binary
│
├── plots/                               # Generated visualizations
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── error_distribution.png
│   └── feature_importance.png
│
├── requirements.txt                     # Python dependencies
└── README.md
```
## Data Source
[https://www.kaggle.com/datasets/thedevastator/optimizing-operating-room-utilization](https://www.kaggle.com/datasets/thedevastator/optimizing-operating-room-utilization)
##  How to Run the Notebook

### 1️⃣ Clone the repository
```bash
git clone [https://github.com/Reddy3001/OR_Utilization-Group-5.git](https://github.com/Reddy3001/OR_Utilization-Group-5.git)
cd OR-Utilization-Project
```
### 2️⃣ Create and activate a virtual environment
It is recommended to use a virtual environment to keep dependencies organized.

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows Git Bash)
source venv/Scripts/activate

# Or for Mac/Linux: 
# source venv/bin/activate
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Launch Jupyter Notebook
```bash
jupyter notebook notebooks/OR_Utilisation_Analysis.ipynb
```
## 📊 Exploratory Data Analysis (EDA)
The notebook performs a deep dive into historical data:
* **Distribution Analysis:** Examining the spread of surgery durations.
* **Variance Analysis:** Workload variation across different OR suites and service lines.
* **Outlier Detection:** Identifying anomalies in case times.
* **Gap Analysis:** Direct comparison of booked vs. actual timing.

---

##  Statistical Testing (A/B Analysis)
We performed hypothesis testing to validate the need for a predictive model by comparing:
* **Group A:** Booked Time (Current baseline scheduling)
* **Group B:** Actual Duration (Ground truth)

**Tests Performed:**
1.  **Paired T-test:** Parametric test for significant differences.
2.  **Wilcoxon Signed-Rank Test:** Non-parametric alternative.

> **Result:** These tests confirmed a statistically significant mismatch between booked and actual durations, validating the requirement for an ML-based approach.

---

##  Machine Learning Models
We trained and evaluated two regression models to predict case duration:
1.  **RandomForestRegressor**
2.  **XGBoostRegressor**

Hyperparameter tuning was performed using `GridSearchCV` (cv=3) optimizing for Negative Mean Absolute Error.

###  Best Model: Random Forest
The **RandomForestRegressor** outperformed XGBoost in stability and error metrics.

**Final Hyperparameters:**
* `max_depth`: 12
* `n_estimators`: 300
* `min_samples_split`: 2

**Evaluation Metrics:**
* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **$R^2$ Score**

---

## 📈 Visualizations
The pipeline automatically generates and saves the following insights to the `/plots` directory:
* **Actual vs. Predicted:** Scatter plot showing correlation.
* **Residual Plot:** Analysis of prediction errors.
* **Error Distribution:** Histogram of error magnitude.
* **Feature Importance:** Which variables drive surgery duration (e.g., Procedure Type, Surgeon, etc.).

---

##  Model Usage
The best-performing model is automatically saved for production use. To load the model in a Python script or API:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/best_or_time_prediction_model.pkl")

# Example prediction
# new_data = pd.DataFrame({...})
# predicted_duration = model.predict(new_data)
```
