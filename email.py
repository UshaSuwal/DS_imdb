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



with st.form(key="my_form"):
    msg=st.text_area("Enter an email to check:")
    data={'Message':[msg]}
    data_df=pd.DataFrame(data)
    predicted=model.predict(data_df["Message"])

    c=st.form_submit_button()

if c:
    st.write("It is ",predicted[0],"message.")

