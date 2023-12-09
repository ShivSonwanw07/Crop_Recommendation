import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load models
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))
meta_model = pickle.load(open('meta_model.pkl', 'rb'))

# Load background image
background_image = 'your_background_image_path_here.jpg'
st.markdown(
    f"""
    <style>
        body {{
            background-image: url('{background_image}');
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Crop Recommendation System 🌾')
st.header('Fill in the required details mentioned below:-')

labels = {'rice': 0, 'maize': 1, 'jute': 2, 'cotton': 3, 'coconut': 4, 'papaya': 5, 'orange': 6,
          'apple': 7, 'muskmelon': 8, 'watermelon': 9, 'grapes': 10, 'mango': 11,
          'banana': 12, 'pomegranate': 13, 'lentil': 14, 'blackgram': 15, 'mungbean': 16,
          'mothbeans': 17, 'pigeonpeas': 18, 'kidneybeans': 19, 'chickpea': 20, 'coffee': 21}

# Function for crop recommendation
def recomm(N, P, K, Temp, Hum, ph, Rain):
    features = pd.DataFrame([[N, P, K, Temp, Hum, ph, Rain]],
                            columns=['N', 'P', 'K', 'Temperature', 'Humidity', 'ph', 'Rainfall'])

    rfc_pred = model1.predict(features)
    dtc_pred = model2.predict(features)
    nbg_pred = model3.predict(features)

    X_meta = np.column_stack((rfc_pred, dtc_pred, nbg_pred))

    pred = meta_model.predict(X_meta)

    for key, value in labels.items():
        if value == pred:
            return key

# Home page content
st.subheader("Welcome to the Crop Recommendation System")
st.write("This system helps you determine the best crops to plant based on environmental parameters.")
st.write("Select an option below:")

# Menu buttons on the home page
menu = st.radio("Select an option", ["Crop Recommendation", "Fertilizer Recommendation"])

if menu == "Crop Recommendation":
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input('Enter Nitrogen(N) levels in the soil:-')
        Temp = st.number_input('Enter Temperature in °C:-')

    with col2:
        P = st.number_input('Enter Phosphorus(P) levels in the soil:-')
        Hum = st.slider('Enter Humidity in %', 0, 100, 50)

    with col3:
        K = st.number_input('Enter Potassium(K) levels in the soil:-')
        ph = st.slider('Enter pH level', 0, 14, 7)

    Rain = st.number_input('Enter Rainfall in mm:-')

    if st.button('Get Recommendation!'):
        recommended_crop = recomm(N, P, K, Temp, Hum, ph, Rain)
        st.success(f"We recommend planting: {recommended_crop}")

elif menu == "Fertilizer Recommendation":
    st.subheader("Fertilizer Recommendation")
    st.write("Fertilizer recommendations for the recommended crop will be displayed here.")
