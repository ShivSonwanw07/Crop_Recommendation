import pickle
import numpy as np
import sklearn
import pandas as pd


# def DecisionTree():
#     dtc=DecisionTreeClassifier()
#     dtc.fit(train_x,train_y)
#     dtc_pred=dtc.predict(test_x)
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
    return pred[0]