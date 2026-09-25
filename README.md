# EduPredict

Predicts a student's exam score (0–100) using machine learning.

## What's inside

```
├── EDA/
│   └── NoteBooks/clean_dataset.ipynb      # cleaning, outliers, encoding
├── Models/
│   ├── LinearRegression.ipynb             # train + 5-fold CV + grid search
│   ├── RandomForestRegressor.ipynb
│   └── SVR.ipynb
├── Data/
│   └── Processed/                         # clean CSVs used by the notebooks
├── streamlit_app/                         # interactive web app for the model
│   ├── app.py
│   ├── linear_regression_model.pkl        # trained model
│   ├── features.pkl                       # column order for prediction
│   ├── requirements.txt
│   └── Dockerfile
└── Test/                                  # scratch files
```

## The app

An interactive Streamlit app that predicts a student's exam score from study habits, background and school context.

Model: **Linear Regression** — best of the 4 compared models. Test metrics: **RMSE 0.33 · MAE 0.27 · R² 0.99**.

For full details on training and evaluation, open the notebooks in `Models/`.

## Run the app locally

```bash
cd streamlit_app
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Run the app with Docker

```bash
docker run -p 8501:8501 selomrani11/edupredict:latest
```

Open http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**.
3. Pick the repository and branch.
4. Set the project directory to `streamlit_app` and the main file to `app.py`.
5. Deploy.