import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    model = joblib.load("linear_regression_model.pkl")
    features = joblib.load("features.pkl")
    return model, features


model, features = load_model()

ORDINAL_OPTS = ["Low", "Medium", "High"]
EDUCATION = {"High School": 1, "College": 2, "Postgraduate": 3}
DISTANCE = {"Near": 1, "Moderate": 2, "Far": 3}

st.title("EduPredict — Exam Score Predictor")
st.markdown("Linear Regression model · RMSE 0.33, R² 0.99")

student = {}

st.subheader("Study habits")
student["Hours_Studied"] = st.slider("Hours Studied", 1, 44, 20)
student["Attendance"] = st.slider("Attendance (%)", 60, 100, 90)
student["Sleep_Hours"] = st.slider("Sleep Hours", 4, 10, 7)
student["Previous_Scores"] = st.slider("Previous Scores", 50, 100, 75)
student["Tutoring_Sessions"] = st.slider("Tutoring Sessions", 0, 8, 2)
student["Physical_Activity"] = st.slider("Physical Activity (hrs/week)", 0, 6, 3)

st.subheader("Parent / student context")
student["Parental_Involvement"] = ORDINAL_OPTS.index(st.selectbox("Parental Involvement", ORDINAL_OPTS)) + 1
student["Access_to_Resources"] = ORDINAL_OPTS.index(st.selectbox("Access to Resources", ORDINAL_OPTS)) + 1
student["Motivation_Level"] = ORDINAL_OPTS.index(st.selectbox("Motivation Level", ORDINAL_OPTS)) + 1
student["Family_Income"] = ORDINAL_OPTS.index(st.selectbox("Family Income", ORDINAL_OPTS)) + 1
student["Teacher_Quality"] = ORDINAL_OPTS.index(st.selectbox("Teacher Quality", ORDINAL_OPTS)) + 1
student["Parental_Education_Level"] = EDUCATION[st.selectbox("Parental Education Level", list(EDUCATION))]
student["Distance_from_Home"] = DISTANCE[st.selectbox("Distance from Home", list(DISTANCE))]
student["Gender"] = st.selectbox("Gender", ["Female", "Male"])
student["School_Type"] = st.selectbox("School Type", ["Public", "Private"])
student["Extracurricular_Activities"] = st.selectbox("Extracurricular Activities", ["No", "Yes"])
student["Internet_Access"] = st.selectbox("Internet Access", ["No", "Yes"])
student["Learning_Disabilities"] = st.selectbox("Learning Disabilities", ["No", "Yes"])
student["Peer_Influence"] = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])

if st.button("Predict Exam Score"):
    row = {col: 0 for col in features}
    for col, val in student.items():
        if col in row:
            row[col] = val
    for col, val in [("Gender", student["Gender"]),
                     ("School_Type", student["School_Type"]),
                     ("Extracurricular_Activities", student["Extracurricular_Activities"]),
                     ("Internet_Access", student["Internet_Access"]),
                     ("Learning_Disabilities", student["Learning_Disabilities"]),
                     ("Peer_Influence", student["Peer_Influence"])]:
        row[f"{col}_{val}"] = 1
    prediction = model.predict(pd.DataFrame([row])[features])[0]
    st.success(f"Predicted Exam Score: {prediction:.1f}")