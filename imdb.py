#!pip install --upgrade scikit-learn

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


@st.cache_data 
def getdata():
    df=pd.read_csv("https://github.com/UshaSuwal/DS_imdb/blob/master/data/cleaned3_imdb.csv",index_col=0)
    return df


st.title("IMDB review Classifications")
df=getdata()




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2,shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ SVM       support vector classifier
])

pipeline.fit(X_train,y_train)


with st.form(key="my_form"):
    msg=st.text_area("Enter a review to check:")
    data={'Message':[msg]}
    data_df=pd.DataFrame(data)
    predicted=pipeline.predict(data_df["Message"])

    c=st.form_submit_button()

if c:
    st.write("It is ",predicted[0],"review.")