from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

input_file = 'preprocessed_data.xlsx'
df = pd.read_excel(input_file)

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the number of estimators
model.fit(X_train, y_train)

joblib.dump(model, 'random_forest_model.pkl')

loaded_model = joblib.load('random_forest_model.pkl')

predictions = loaded_model.predict(X_test)

print(predictions)
