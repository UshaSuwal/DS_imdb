# regression for multivariate
import numpy as np
import pandas as pd
import streamlit as st


st.title("Salary Prediction")

@st.cache_data
def getdata():
    df=pd.read_csv("cleaned_salary.csv",index_col=0)
    return df

df=getdata()






from sklearn.model_selection import train_test_split
# We specify this so that the train and test data set always have the same rows, respectively
df_train, df_test = train_test_split(df, train_size = 0.85, test_size = 0.15, random_state = 1)

from sklearn.linear_model import LinearRegression
X_train = df_train[['Age','Years of Experience',"Bachelor's",           # x_train is independent variable
                    "Master's","PhD"]]

y_train = df_train['Salary']  #y_train is dependent variable
# df_test_predict = X_train.iloc[0:1]

X_test=df_test[['Age','Years of Experience',"Bachelor's",           # x_train is independent variable
                    "Master's","PhD"]]
y_test=df_test['Salary']
lm = LinearRegression()
model = lm.fit(X_train, y_train)



with st.form(key="my_form"):
    age=st.number_input("Enter your age:",min_value=18)
    experience=st.number_input("Enter year of experience:",min_value=0)
    education=st.radio("Education Level", ["Bachelor's", "Master's","PhD"])

    if education=="Bachelor's":
        b=1
        m=0
        p=0
    elif education=="Master's":
        m=1
        b=0
        p=0
    elif education=="Phd":
        m=0
        b=0
        p=1
    else:
        st.write("Please select one ")
    c=st.form_submit_button("Analyze")




    data={'Age':[age],
          'Years of Experience':[experience],
          "Bachelor's":[b],
          "Master's":[m],
          "PhD":[p]};
    data_df=pd.DataFrame(data)
    predicted=model.predict(data_df)

    
if c:
    st.write("Predicted Salary: Rs ",int(predicted[0]))


