#!pip install --upgrade scikit-learn

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


@st.cache_data 
def getdata():
    df=pd.read_csv("cleaned3_imdb.csv",index_col=0)
    return df


st.title("IMDB review Analyzer")
df=getdata()


st.header("Model used:")
st.write("Support Vector Machine.")
st.write("The IMDb Review Analyzer using SVM is a machine learning model designed for sentiment analysis of movie reviews from the IMDb dataset. Leveraging the Support Vector Machine (SVM) algorithm, the model classifies reviews into positive or negative sentiments based on the textual content. Trained on a labeled dataset, it learns to discern sentiment patterns, allowing users to quickly assess the overall tone of IMDb movie reviews.")
st.header("Analyze:")


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
predictions=pipeline.predict(X_test)


with st.form(key="my_form"):
    msg=st.text_area("Enter a review to check:")
    data={'Message':[msg]}
    data_df=pd.DataFrame(data)
    predicted=pipeline.predict(data_df["Message"])

    c=st.form_submit_button()

if c:
    st.write("It is ",predicted[0],"review.")




st.header("Model Accuracy Score:")
from sklearn.metrics import accuracy_score

# Assuming Y_test is the actual labels
# and predicted_risk contains the predicted labels
accuracy = accuracy_score(y_test,predictions)
st.write(f'Accuracy: {accuracy * 100:.2f}%')



st.header("Classification Report:")
from sklearn.metrics import classification_report

# Assuming Y_test is the actual labels
# and predicted_risk contains the predicted labels
report = classification_report(y_test, predictions)

st.text("Classification Report:\n{}".format(report))



st.header("Confusion Matrix:")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Create a confusion matrix
cm = confusion_matrix(pipeline.predict(X_test), y_test)

# Change figure size and increase dpi for better resolution
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
class_names = ['negative', 'positive']
display = ConfusionMatrixDisplay(cm, display_labels=class_names)

# Show the plot. Pass the parameter ax to show customizations (ex. title)
display.plot(ax=ax)
st.pyplot(fig)