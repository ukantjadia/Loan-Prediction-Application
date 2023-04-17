import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.beta_set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation  🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
        with st.beta_expander(" ℹ️ Information", expanded=True):
            st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.

            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''


# attributes are 
    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        gender = st.radio("Gender", ['Male','Female'])
        married = st.radio("Married", ['Yes','No'])
        dependents = st.radio("Dependents", ['0','1','2','3'])
        education = st.radio("Education", ['Graduate','Not Graduate'])
        employed = st.radio("Self Employed", ['Yes','No'])
        propertyArea = st.radio("Property Area", ['Semiurban','Urban'])
        applicantIncome = st.number_input("Applicant Income", 150.0,100000.0)
        coApplicantIncome = st.number_input("Co Applicant Income", 0.0,50000.0)
        loanAmount = st.number_input("Loan Amount", 10.0,1000.0)
        loanAmountTerm = st.number_input("Loan Amount Term", 10.0,600.0)
        creditHistory = st.number_input("Credit History", 0.0,1.0)

        feature_list = [gender, married, dependents, education, employed, propertyArea, applicantIncome, coApplicantIncome, loanAmount, loanAmountTerm, creditHistory]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
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