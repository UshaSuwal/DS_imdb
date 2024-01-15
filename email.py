#pip install streamlit
#pip install -U scikit-learn

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import streamlit as st


@st.cache_data 
def getdata():
    df=pd.read_csv("cleaned2_email.csv",index_col=0)
    return df


df=getdata()
df = df.dropna(subset=['Message'])


st.title("Spam Email Detection")

st.header("Model Used:")
st.write("Logistic Regression")
st.write("The Spam Email Classifier using Logistic Regression is a machine learning model designed to identify whether an email is legitimate (ham) or spam. Leveraging logistic regression, a binary classification algorithm, the model has been trained on a dataset of labeled emails. By analyzing various features within the email content, such as keywords. The model learns to distinguish between legitimate and spam emails.")
st.header("Analyze:")
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer() 
# Split the data
X = df['Message']
Y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

# Build and train the model
p = Pipeline([('vect', vectorizer),
              ('chi', SelectKBest(chi2, k=2000)),
              ('clf', LogisticRegression())])

model = p.fit(X_train, y_train)
prediction=model.predict(X_test)



with st.form(key="my_form"):
    msg=st.text_area("Enter an email to check:")
    data={'Message':[msg]}
    data_df=pd.DataFrame(data)
    predicted=model.predict(data_df["Message"])

    c=st.form_submit_button()

if c:
    st.write("It is ",predicted[0],"message.")




st.header("Model Accuracy Score:")
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,prediction)
st.write(f'Accuracy: {accuracy * 100:.2f}%')


st.header("Classification Report:")
from sklearn.metrics import classification_report
report = classification_report(y_test, prediction)
st.text("Classification Report:\n{}".format(report))


st.header("Confusion Matrix:")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(model.predict(X_test), y_test)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
class_names = ['ham', 'spam']
display = ConfusionMatrixDisplay(cm, display_labels=class_names)
display.plot(ax=ax)
st.pyplot(fig)