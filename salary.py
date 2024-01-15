# regression for multivariate
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.title("Salary Prediction")

@st.cache_data
def getdata():
    df=pd.read_csv("cleaned_salary.csv",index_col=0)
    return df

df=getdata()


st.header("Model used:")
st.write("Linear Regression")
st.write("This salary prediction model utilizes linear regression to estimate salaries for individuals based on key features like years of experience, age, and education level. By leveraging historical data that includes known salaries and corresponding feature values, the model learns the underlying patterns and correlations. Through this learning process, the model develops a linear equation that best fits the relationship between the input features and the target variable (salary).")

st.header("Analyze:")
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
predicted_salary=model.predict(X_test)



with st.form(key="my_form"):
    age=st.number_input("Enter your age:",min_value=20)
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
    elif education=="PhD":
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


st.header("Accuracy Score of Model:")
from sklearn.metrics import r2_score

# Assuming y_test is the actual labels and predicted_salary contains the predicted labels
r2 = r2_score(y_test, predicted_salary)
accuracy_percentage = r2 * 100
st.write(f'Accuracy: {accuracy_percentage:.2f}%')



st.set_option('deprecation.showPyplotGlobalUse', False)

st.header("Actual vs Predicted Salary on Training Set")
plt.scatter(y_train, model.predict(X_train), color='red')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary on Training Set')

# Use Streamlit to display the Matplotlib figure with st.pyplot(fig)
st.pyplot()



