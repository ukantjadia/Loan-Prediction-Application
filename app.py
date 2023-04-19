import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import joblib

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def load_model(modelfile):
    loaded_model = joblib.load(modelfile)
    return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Loan Prediction 💴🏦 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.write(""" ## How does it work ❓
The loan approval prediction model uses your credit score, income, education, loan amount and other various factor  to determine whether your loan will be approved or not.  It outputs a "yes" or "no" decision.
                """)
    

    with st.sidebar.expander(" ℹ️ Information", expanded=True):
        st.write("""
                Loan Prediction is very helpful for employee of banks 
                as well as for the applicant also.
                Dream housing Finance Company 
                deals in all loans. They have presence across all urban, 
                semi urban and rural areas. Customer first apply for loan 
                after that company or bank validates the customer 
                eligibility for loan. Company or bank wants to automate 
                the loan eligibility process (real time) based on customer 
                details provided while filling application form. These 
                details are Gender, Marital Status, Education, Number of 
                Dependents, Income, Loan Amount, Credit History and 
                other
        """)



    with st.form("my_form"):
        col4,col6 = st.columns([2,2])
        with col4:
            education = st.radio("Education", ['Graduate','Not Graduate'])
            married = st.radio("Married", ['Yes','No'])
            loanAmountTerm = st.number_input("Loan Amount Term", 10.0,600.0)
            applicantIncome = st.number_input("Applicant Income", 150.0,100000.0)
            propertyArea = st.radio("Property Area", ['Ruler','Semiurban','Urban'])
        with col6:
            employed = st.radio("Self Employed", ['Yes','No'])
            gender = st.radio("Gender", ['Male','Female'])
            coApplicantIncome = st.number_input("Co Applicant Income", 0.0,50000.0)
            loanAmount = st.number_input("Loan Amount", 10.0,1000.0)
            creditHistory = str(st.slider("Credit History",1,3000))
            dependents = st.slider("Dependents", 0,4)

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            num_data = [applicantIncome, coApplicantIncome, loanAmount, loanAmountTerm]
            cat_data = [gender, married, dependents, education, employed,creditHistory, propertyArea]
            test=[]
            test = list(le.fit_transform(cat_data))
            feature_list = num_data + test
            single_pred = np.array(feature_list).reshape(1,-1)
        
            loaded_model = load_model('model.sav')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            if prediction[0] == "0":
                st.write("Yes")
            else: 
                st.write("No")

        #     col1.success(f"{prediction} are recommended by the A.I for your farm.")
      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()