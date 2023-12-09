import streamlit as st
import pandas as pd 
import pickle
import numpy as np
import sklearn


st.title('Crop Recommendation System 🌾')
st.header('Fill the required details mentioned below:-')
labels={
    'rice':0,'maize':1,'jute':2,'cotton':3,'coconut':4,'papaya':5,'orange':6,
    'apple':7,'muskmelon':8,'watermelon':9,'grapes':10,'mango':11,
    'banana':12,'pomegranate':13,'lentil':14,'blackgram':15,'mungbean':16,
    'mothbeans':17,'pigeonpeas':18,'kidneybeans':19,'chickpea':20,'coffee':21
}

model1=pickle.load(open('model1.pkl','rb'))
model2=pickle.load(open('model2.pkl','rb'))
model3=pickle.load(open('model3.pkl','rb'))
meta_model=pickle.load(open('meta_model.pkl','rb'))


def recomm(N,P,K,Temp,Hum,ph,Rain):
#   features=np.array([[N,P,K,Temp,Hum,ph,Rain]])
    features =pd.DataFrame([[N, P, K, Temp, Hum, ph, Rain]], columns=['N', 'P', 'K', 'Temperature', 'Humidity', 'ph', 'Rainfall'])

    rfc_pred=model1.predict(features)
    dtc_pred=model2.predict(features)
    nbg_pred=model3.predict(features)

    X_meta = np.column_stack((rfc_pred,dtc_pred, nbg_pred))
    
    pred=meta_model.predict(X_meta)

    for key, value in labels.items():
        if value == pred:
            return key
        



col1,col2,col3= st.columns(3)

with col1:
    N=st.number_input('Enter the Nitrogen(N) levels in the soil:-')
    Temp=st.number_input('Enter the Temperature in °C:-')

with col2:
    P=st.number_input('Enter the Phosphorus(P) levels in the soil:-')
    Hum = st.slider('Enter the Humidity in %', 0, 100, 50)

with col3:
     K=st.number_input('Enter the Potassium(K) levels in the soil:-')
     ph = st.slider('Enter the pH level', 0, 14, 7)

Rain=st.number_input('Enter the Rainfall in mm:-')

if st.button('GET RECOMMENDATION!'):
    a=recomm(N,P,K,Temp,Hum,ph,Rain)
    st.header(a)