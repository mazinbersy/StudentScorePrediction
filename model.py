import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data.dropna(axis=0)

def split_data(data, features, target, test_size=0.2, random_state=1):
    x = data[features]
    y = data[target]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def train_linear_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def plot_linear_results(x_test, y_test, y_pred):
    plt.scatter(x_test, y_test, color='blue', label='Actual')
    plt.plot(x_test, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.legend()
    plt.show()

def train_polynomial_model(x_train, y_train, degree):
    poly = PolynomialFeatures(degree=degree)
    x_poly_train = poly.fit_transform(pd.DataFrame(x_train, columns=["Hours_Studied"]))
    model = LinearRegression()
    model.fit(x_poly_train, y_train)
    return poly, model

def plot_polynomial_results(x, y, poly, model, degree):
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    x_range_poly = poly.transform(pd.DataFrame(x_range, columns=["Hours_Studied"]))
    y_range_pred = model.predict(x_range_poly)

    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x_range, y_range_pred, color='red', label=f'Polynomial Degree {degree}: Actual vs Predicted')
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Polynomial Regression Fit")
    plt.legend()
    plt.show()

def print_metrics(linear, poly):
    linear_mae, linear_mse, linear_rmse, linear_r2 = linear
    poly_mae, poly_mse, poly_rmse, poly_r2 = poly

    table = [["Mean Absolute Error (MAE)", f"{linear_mae:.3f}", f"{poly_mae:.3f}"],
             ["Mean Squared Error (MSE)", f"{linear_mse:.3f}", f"{poly_mse:.3f}"],
             ["Root Mean Squared Error (RMSE)", f"{linear_rmse:.3f}", f"{poly_rmse:.3f}"],
             ["R² Score", f"{linear_r2:.3f}", f"{poly_r2:.3f}"]]

    print(tabulate(table, headers=["Metric", "Linear Regression", "Polynomial Regression"], tablefmt = "grid"))

def main():
    filepath = "StudentPerformanceFactors.csv"
    data = load_data(filepath)

    features = ["Hours_Studied"]
    target = "Exam_Score"
    x_train, x_test, y_train, y_test = split_data(data, features, target)

    # Linear Regression
    linear_model = train_linear_model(x_train, y_train)
    y_pred_linear = linear_model.predict(x_test)
    linear_metrics = evaluate_model(y_test, y_pred_linear)
    plot_linear_results(x_test, y_test, y_pred_linear)

    # Polynomial Regression
    degree = 3
    poly_transformer, poly_model = train_polynomial_model(x_train, y_train, degree)
    x_poly_test = poly_transformer.transform(pd.DataFrame(x_test, columns=["Hours_Studied"]))
    y_pred_poly = poly_model.predict(x_poly_test)
    poly_metrics = evaluate_model(y_test, y_pred_poly)
    plot_polynomial_results(x_test, y_test, poly_transformer, poly_model, degree)

    print_metrics(linear_metrics, poly_metrics)

if __name__ == "__main__":
    main()
