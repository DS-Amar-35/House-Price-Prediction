import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import joblib

df=pd.read_csv(r"C:\Users\jakku\Downloads\House_price.csv")
print(df.head())


print(df.isnull().sum())
print(df.describe())

X=df[['Bedrooms','Bathrooms','Age_years']]
y=df['Price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=LinearRegression()
model.fit(X_train,y_train)

pred=model.predict(X_test)
print("R2:",r2_score(y_test,pred))

joblib.dump(model,"linear_regression_model.pkl")


import joblib
import pandas as pd

model=joblib.load("linear_regression_model.pkl")
sample=[[5000,3,2]]
print(model.predict(sample))


      
      

