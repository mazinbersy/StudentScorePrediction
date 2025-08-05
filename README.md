# 📊 Student Performance Predictor

This project implements **Linear Regression** and **Polynomial Regression** models to predict student exam scores based on the number of hours studied. The goal is to visualize and compare the performance of both models using various evaluation metrics.

---

## 🚀 Features

- Loads and preprocesses real-world student data
- Splits dataset into training and testing sets
- Trains:
  - Linear Regression model
  - Polynomial Regression model (configurable degree)
- Evaluates both models using:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
- Visualizes:
  - Linear regression fit
  - Polynomial regression curve
- Tabulates and compares model performance using `tabulate`

---

## 🧠 Model Comparison Example

| Metric                     | Linear Regression | Polynomial Regression |
|---------------------------|-------------------|------------------------|
| Mean Absolute Error (MAE) |        5.412       |         3.298          |
| Mean Squared Error (MSE)  |       40.152       |         19.301         |
| Root Mean Squared Error   |        6.335       |         4.393          |
| R² Score                  |        0.821       |         0.924          |

*Note: Values shown are examples. Actual output depends on the dataset.*

---

## 📁 Dataset

The code uses a CSV file: `StudentPerformanceFactors.csv`, which should contain at least the following columns:

- `Hours_Studied`
- `Exam_Score`

Make sure the CSV is clean (no missing values) and placed in the same directory as the script.

---

## 📈 Output
The program will:
- Display two graphs:
  - Linear Regression Line
  - Polynomial Regression Curve
- Print a comparison table of evaluation metrics

---

## 🛠️ Requirements

Install dependencies with:
```bash
pip install pandas numpy matplotlib scikit-learn tabulate
```



