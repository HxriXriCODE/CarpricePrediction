import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from tabulate import tabulate
import scipy.stats as stats
from sklearn.model_selection import train_test_split , KFold
import warnings

df=pd.read_csv(r"C:\Users\harik\Downloads\archive\car_price.csv")
print(df.head(5))
print(df.tail(5))
df.shape
df.info()
#.T to transpoce and bar chart
df.describe().T.plot(kind='bar')
# this to rel btw engsize to price
sns.scatterplot(x='enginesize', y='price', data=df)
# due to 147 uq carname so I droped it 
df_temp = df.drop('CarName', axis=1)
df_encoded = pd.get_dummies(df_temp, drop_first=True)
# encoding everything with values 
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make new predictions
rf_preds = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, rf_preds)
r2 = r2_score(y_test, rf_preds)

print(f"Average Error: ${mae:.2f}") # Average Error: $1865.70
print(f"R-squared Score: {r2:.2f}") # R-squared Score: 0.92
