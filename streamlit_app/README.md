# EduPredict — Exam Score Predictor App

Interactive [Streamlit](https://streamlit.io) app that predicts a student's exam score (0–100) from their study habits, background and school context.

Model: **Linear Regression** (the best of the 4 models compared in the notebooks). Metrics on the held-out test set: **RMSE 0.33 · MAE 0.27 · R² 0.99**.

## How it works

- The UI collects 19 inputs (study habits, parent context, school info).
- Those inputs are encoded the same way as in `EDA/NoteBooks/clean_dataset.ipynb` (ordinal values → 1/2/3, categorical values → one-hot).
- The trained model (`linear_regression_model.pkl`, trained in `Models/LinearRegression.ipynb`) predicts the score.

## Folder layout

```
streamlit_app/
├── app.py                        # the Streamlit UI (entrypoint)
├── linear_regression_model.pkl   # trained model
├── features.pkl                  # column order used for prediction
├── requirements.txt
└── Dockerfile
```

## Run it locally

```bash
cd streamlit_app
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Run it with Docker

Build:

```bash
docker build -t selomrani11/edupredict:latest streamlit_app/
```

Run:

```bash
docker run -p 8501:8501 selomrani11/edupredict:latest
```

Open http://localhost:8501.

## Push the image to Docker Hub

```bash
docker login                        # if not already logged in
docker push selomrani11/edupredict:latest
```

Anyone can then run it with: `docker run -p 8501:8501 selomrani11/edupredict:latest`

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**.
3. Pick the repository + branch.
4. Set the project directory to `streamlit_app` and the main file to `app.py`.
5. Deploy. Requirements install automatically from `requirements.txt`.

## Rebuild the model (optional)

The committed `.pkl` files are the source of truth and are enough to run the app. To retrain from scratch:

```bash
cd ..   # repo root
uv pip install -r streamlit_app/requirements.txt
python - <<'EOF'
import joblib, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/Processed/clean_data_encoded.csv")
X = df.drop(columns=["Exam_Score"]); y = df["Exam_Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression(); model.fit(X_train, y_train)

joblib.dump(model, "streamlit_app/linear_regression_model.pkl")
joblib.dump(list(X.columns), "streamlit_app/features.pkl")
EOF
```

## Related notebooks

| File | Purpose |
|---|---|
| `EDA/NoteBooks/clean_dataset.ipynb` | Cleaning, IQR outliers, encoding |
| `Models/LinearRegression.ipynb` | Train + 5-fold CV + grid search |
| `Models/RandomForestRegressor.ipynb` | Same for Random Forest |
| `Models/SVR.ipynb` | Same for SVR |