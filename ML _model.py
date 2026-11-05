import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import math
df=pd.read_csv(r"C:\Users\jakku\Downloads\House_price.csv")
X =df[["Size_sqft", "Bedrooms", "Bathrooms", "Age_years"]]
y =df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

