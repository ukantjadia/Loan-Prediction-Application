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

    col1,col2  = st.columns([2,2])
    
    with col1: 
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

            gender = st.radio("Gender", ['Male','Female'])
            married = st.radio("Married", ['Yes','No'])
            dependents = st.radio("Dependents", ['0','1','2','3'])


# attributes are 
    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        col3,col4  = st.columns([2,2])
        with col3:
            applicantIncome = st.number_input("Applicant Income", 150.0,100000.0)
            coApplicantIncome = st.number_input("Co Applicant Income", 0.0,50000.0)
            loanAmount = st.number_input("Loan Amount", 10.0,1000.0)
        with col4:
            education = st.radio("Education", ['Graduate','Not Graduate'])
            employed = st.radio("Self Employed", ['Yes','No'])
            propertyArea = st.radio("Property Area", ['Semiurban','Urban'])
            loanAmountTerm = st.number_input("Loan Amount Term", 10.0,600.0)
            creditHistory = str(st.slider("Credit History",1,3000))

            num_data = [applicantIncome, coApplicantIncome, loanAmount, loanAmountTerm]
            cat_data = [gender, married, dependents, education, employed,creditHistory, propertyArea]
            test=[]
            test = le.fit_transform(cat_data)

        feature_list = num_data + cat_data
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):
            loaded_model = load_model('model.sav')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"{prediction.item().title()} are recommended by the A.I for your farm.")
      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon. Check the source code [here](https://github.com/gabbygab1233/Crop-Recommendation)")
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