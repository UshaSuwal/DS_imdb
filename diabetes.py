# regression for multivariate
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


st.title("Diabetes Prediction")

@st.cache_data
def getdata():
    data=pd.read_csv("cleaned_dibetic.csv",index_col=0)
    return data

data=getdata()


st.header("Model used:")
st.write("k-Nearest Neighbors")
st.write("The Diabetes Prediction Model using KNN is a machine learning model designed to assess the likelihood of an individual having diabetes based on relevant features such as age, BMI, blood pressure, and other health indicators. KNN, or k-Nearest Neighbors, is employed to classify a new data point by considering the majority class among its k nearest neighbors in the feature space.")

st.header("Analyze:")


X = data.iloc[:,:8]
Y = data.iloc[:,8:]
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state=1)
from sklearn.neighbors import KNeighborsClassifier
KNN= KNeighborsClassifier(n_neighbors=14)
KNN.fit(X_train,y_train)
predicted= KNN.predict(X_test)





with st.form(key="my_form"):
    age=st.number_input("Enter your age:",min_value=0)
    preg=st.number_input("Enter number of pregnancies:",min_value=0)
    insulin=st.number_input("Insulin:",min_value=0.0)
    glucose=st.number_input("Glucose:",min_value=0.0)
    pressure=st.number_input("Blood Pressure:",min_value=0.0)
    skin=st.number_input("Skin Thickness:",min_value=0.0)
    bmi=st.number_input("BMI:",min_value=0.0)
    func=st.number_input("Dibetes Pedidree Function:",min_value=0.0)

    c=st.form_submit_button("Analyze")




    data={'Pregnancies':[preg],
          'Glucose':[glucose],
          "BloodPressure":[pressure],
          "SkinThickness":[skin],
          "Insulin":[insulin],
          "BMI":[bmi],
          "DiabetesPedigreeFunction":[func],
          "Age":[age],
          };
    data_df=pd.DataFrame(data)
    prediction=KNN.predict(data_df)

    
if c:
    if prediction==1:
        st.write("Have Diabetes")
    elif prediction==0:
        st.write("No Diabetes")

    


st.header("Accuracy Score of Model:")
from sklearn.metrics import accuracy_score

# Assuming Y_test is the actual labels
# and predicted_risk contains the predicted labels
accuracy = accuracy_score(y_test,predicted)
st.write(f'Accuracy: {accuracy * 100:.2f}%')



st.header("Classification Report:")
from sklearn.metrics import classification_report
report = classification_report(y_test, predicted)
st.text("Classification Report:\n{}".format(report))


st.header("Confusion Matrix:")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(KNN.predict(X_test), y_test)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
class_names = ['0', '1']
display = ConfusionMatrixDisplay(cm, display_labels=class_names)
display.plot(ax=ax)
st.pyplot(fig)



