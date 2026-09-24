import joblib
import os
import pandas as pd
import streamlit as st

HERE = os.path.dirname(__file__)

model = joblib.load(os.path.join(HERE, "linear_regression_model.pkl"))
features = joblib.load(os.path.join(HERE, "features.pkl"))

st.title("Exam Score Predictor (Linear Regression)")

student = {}

st.subheader("Study habits")
student["Hours_Studied"] = st.slider("Hours Studied", 1, 44, 20)
student["Attendance"] = st.slider("Attendance (%)", 60, 100, 90)
student["Sleep_Hours"] = st.slider("Sleep Hours", 4, 10, 7)
student["Previous_Scores"] = st.slider("Previous Scores", 50, 100, 75)
student["Tutoring_Sessions"] = st.slider("Tutoring Sessions", 0, 8, 2)
student["Physical_Activity"] = st.slider("Physical Activity (hrs/week)", 0, 6, 3)

st.subheader("Parent / student context")
ordinal_opts = ["Low", "Medium", "High"]
student["Parental_Involvement"] = ordinal_opts.index(st.selectbox("Parental Involvement", ordinal_opts)) + 1
student["Access_to_Resources"] = ordinal_opts.index(st.selectbox("Access to Resources", ordinal_opts)) + 1
student["Motivation_Level"] = ordinal_opts.index(st.selectbox("Motivation Level", ordinal_opts)) + 1
student["Family_Income"] = ordinal_opts.index(st.selectbox("Family Income", ordinal_opts)) + 1
student["Teacher_Quality"] = ordinal_opts.index(st.selectbox("Teacher Quality", ordinal_opts)) + 1
edu = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
student["Parental_Education_Level"] = {"High School": 1, "College": 2, "Postgraduate": 3}[edu]
dist = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
student["Distance_from_Home"] = {"Near": 1, "Moderate": 2, "Far": 3}[dist]

st.subheader("Nominal categories")
student["Gender"] = st.selectbox("Gender", ["Female", "Male"])
student["School_Type"] = st.selectbox("School Type", ["Public", "Private"])
student["Extracurricular_Activities"] = st.selectbox("Extracurricular Activities", ["No", "Yes"])
student["Internet_Access"] = st.selectbox("Internet Access", ["No", "Yes"])
student["Learning_Disabilities"] = st.selectbox("Learning Disabilities", ["No", "Yes"])
student["Peer_Influence"] = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])

if st.button("Predict Exam Score"):
    row = {col: 0 for col in features}
    student_encoded = student.copy()
    for col, val in student_encoded.items():
        if col in row:
            row[col] = val
    for prefix, val in [("Gender", student["Gender"]),
                        ("School_Type", student["School_Type"]),
                        ("Extracurricular_Activities", student["Extracurricular_Activities"]),
                        ("Internet_Access", student["Internet_Access"]),
                        ("Learning_Disabilities", student["Learning_Disabilities"]),
                        ("Peer_Influence", student["Peer_Influence"])]:
        row[f"{prefix}_{val}"] = 1
    X_new = pd.DataFrame([row])[features]
    pred = model.predict(X_new)[0]
    st.success(f"Predicted Exam Score: {pred:.1f}")