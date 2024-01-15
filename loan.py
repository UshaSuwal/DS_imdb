
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import streamlit as st






@st.cache_data 
def getdata():
    df=pd.read_csv("cleaned1_loan.csv",index_col=0)
    return df
df=getdata()




# We specify this so that the train and test data set always have the same rows, respectively
df_train, df_test = train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 0)
X_train = df_train[["cibil_score", "Assets_to_loan_score", "Income_to_Loan_score_per_dependancy"]]
y_train = df_train["loan_status"]
X_test  = df_test[["cibil_score", "Assets_to_loan_score", "Income_to_Loan_score_per_dependancy"]]
y_test = df_test["loan_status"]
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()    #model
model.fit(X_train,y_train)
predict_target = model.predict(X_test)    


st.title("Loan Approval Prediction")

st.header("Model used:")
st.write("Logistic Regression")
st.write("This loan prediction model utilizes logistic regression to evaluate loan applications, providing a outcome of either approval or rejection. It analyzes a set of input features, which could include factors such as the applicant's credit score, income, assets value, and other relevant financial information. By learning from historical data where loan outcomes are known, the logistic regression model establishes a relationship between these features and the likelihood of loan approval.")


with st.form(key="my_form"):
   gender=st.radio("Gender", ["Male", "Female"])
   education=st.radio("Education", ["Graduate", "Ungraduate"])
   residential_assets= st.number_input("Enter residential assets value")
   commercial_assets= st.number_input("Enter commercial_assets_value")
   luxury_assets= st.number_input("Enter luxury_assets_value")
   cibil= st.number_input("Enter cibil score",300,900)
   loan_amt=st.number_input("Enter loan amount",min_value=10000)
   income=st.number_input("Enter annual income")
   loanTerm=st.number_input("Enter loan term")
   dependents=st.number_input("Enter number of dependent")

   if dependents==0:
       dependents=0.001
   

   TotalAssets=residential_assets+commercial_assets+luxury_assets
   AssetsScore = TotalAssets*cibil
   Assets_loan_score=AssetsScore/loan_amt
   IncomeLoan=(income* loanTerm) / loan_amt
   IncomeLoan_score =IncomeLoan *cibil
   IncomeLoan_per_dependancy= IncomeLoan_score/dependents


   details={"cibil_score":[cibil],
            "Assets_to_loan_score":[Assets_loan_score],
            "Income_to_Loan_score_per_dependancy":[IncomeLoan_per_dependancy]};
   details_df=pd.DataFrame(details);
   predict=model.predict(details_df)
   c=st.form_submit_button()

if c:
    if predict==1:
        st.write("Loan Approved!")
    else:
        st.write("Loan Rejected!")



st.header("Accuracy Score of Model:")
from sklearn.metrics import accuracy_score

# Assuming Y_test is the actual labels
# and predicted_risk contains the predicted labels
accuracy = accuracy_score(y_test,predict_target)
st.write(f'Accuracy: {accuracy * 100:.2f}%')



st.header("Classification Report:")
from sklearn.metrics import classification_report

# Assuming Y_test is the actual labels
# and predicted_risk contains the predicted labels
report = classification_report(y_test, predict_target)

st.text("Classification Report:\n{}".format(report))



st.header("Confusion Matrix:")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assuming X_test and y_test are your test data
cm = confusion_matrix(model.predict(X_test), y_test)

# Change figure size and increase dpi for better resolution
# and get reference to axes object
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
class_names = ['0', '1']

# initialize using the raw 2D confusion matrix
# and output labels (in our case, it's 0 and 1)
display = ConfusionMatrixDisplay(cm, display_labels=class_names)

# set the plot title using the axes object

ax.grid(False)

# show the plot.
# Pass the parameter ax to show customizations (ex. title)
display.plot(ax=ax)

# Use Streamlit to display the Matplotlib figure
st.pyplot(fig)
