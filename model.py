import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import joblib

# Load the dataset from a CSV file
data = pd.read_csv("DBS_SingDollar.csv")

# Separate features (SGD) and target (DBS)
X = data[['SGD']]
y = data[['DBS']]

# Initialize and train the linear regression model
regressor = LinearRegression()
regressor.fit(X, y)

# Predict target values using the trained model
predictions = regressor.predict(X)

# Compute the Root Mean Squared Error (RMSE)
rmse = root_mean_squared_error(y, predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Save the trained model to a file
joblib.dump(regressor, 'dbs_linear_model.pkl')

# Load the model from the file
loaded_regressor = joblib.load('dbs_linear_model.pkl')

# Display the model's coefficients and intercept
print(f"Model Coefficients: {loaded_regressor.coef_}")
print(f"Model Intercept: {loaded_regressor.intercept_}")
