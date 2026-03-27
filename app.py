import streamlit as st
import numpy as np
import pickle
model=pickle.load(open("heart_disease.pkl", 'rb'))
scaler=pickle.load(open("scaler.pkl",'rb'))

st.title("Heart Disease Prediction")
st.write("Fill the Following details to predict if a person have disease or not")

age=st.number_input("Age", 20, 100,50)
sex=st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
cp=st.selectbox("Chest Pain Type(cp)", [0,1,2,3])
trestbps=st.number_input("Resting Blood Presssure (trestbps)", 80, 200,120)
chol=st.number_input("Cholesterol (chol)", 100, 600, 200)
fbs=st.selectbox("Fasting Blood Sugar>120 mg/dl(fbs)", [0,1])
restcg=st.selectbox("Resting ECG(restcg)", [0,1,2])
thalach=st.number_input("Max Heart Rate Achived (thalach)", 60, 250,150)
exang=st.selectbox("Exercise Include Angina (exang)", [0,1])
oldpeak=st.number_input("ST depression (oldpeak)", 0.0,6.0,1.0)
slope=st.selectbox("Slope of ST (slope)", [0,1,2])
ca=st.selectbox("Number of Majaor Vessels (ca)", [0,1,2,3,4])
thal=st.selectbox("Thalassemia (thal)", [0,1,2,3])

if st.button("Predict"):
    input_data=np.array([[age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data_scaled=scaler.transform(input_data)
    prediction=model.predict(input_data_scaled)
    if prediction[0]==1:                            
        st.success("The person is likely to have heart disease.")                                                                     
    else:
        st.success("The person is unlikely to have heart disease.")
