import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

input_file = 'preprocessed_data.xlsx'
df = pd.read_excel(input_file)

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_performance = {}

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_performance[model_name] = {"MSE": mse, "MAE": mae, "R^2": r2}

best_model_name = min(model_performance, key=lambda k: model_performance[k]["MSE"])
best_model = models[best_model_name]

for model_name, metrics in model_performance.items():
    print(f"{model_name}:")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"R-squared (R^2): {metrics['R^2']:.4f}")
    print("\n")

print(f"The best model is {best_model_name} with an MSE of {model_performance[best_model_name]['MSE']:.4f}")
