import joblib
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(HERE, "../Data/Processed/clean_data_encoded.csv"))
X = df.drop(columns=["Exam_Score"])
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, os.path.join(HERE, "linear_regression_model.pkl"))
joblib.dump(list(X.columns), os.path.join(HERE, "features.pkl"))