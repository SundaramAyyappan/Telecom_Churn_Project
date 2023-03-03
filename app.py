# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:29:39 2023

@author: Jeevika
"""

import numpy as np
import pickle
import pandas as pd

import streamlit as st 

from PIL import Image

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_note_authentication(voice_plan_yes, voice_messages, intl_plan_yes, intl_mins, intl_calls, intl_charge, day_mins, day_calls, day_charge, eve_mins, eve_calls, eve_charge, night_mins, night_calls, night_charge, customer_calls):
    
    prediction=classifier.predict([[voice_plan_yes, voice_messages, intl_plan_yes, intl_mins, intl_calls, intl_charge, day_mins, day_calls, day_charge, eve_mins, eve_calls, eve_charge, night_mins, night_calls, night_charge, customer_calls]])
   print(prediction)
   return prediction



def main():
    
    st.title('Churn prediction Web App')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    voice_plan_yes = st.sidebar.selectbox('Voiceplan',('1','0'))
    voice_messages = st.sidebar.number_input('Insert no of voice_message')
    intl_plan_yes = st.sidebar.selectbox('Intlplan',('1','0'))
    intl_mins = st.sidebar.number_input('Insert intl mins')
    intl_calls = st.sidebar.number_input('Insert intl calls')
    intl_charge = st.sidebar.number_input('Insert intl charge')
    day_mins = st.sidebar.number_input('Insert day mins')
    day_calls = st.sidebar.number_input('Insert day calls')
    day_charge = st.sidebar.number_input('Insert day charge')
    eve_mins = st.sidebar.number_input('Insert eve mins')
    eve_calls = st.sidebar.number_input('Insert eve calls')
    eve_charge = st.sidebar.number_input('Insert eve charge')
    night_mins = st.sidebar.number_input('Insert night mins')
    night_calls = st.sidebar.number_input('Insert night calls')
    night_charge = st.sidebar.number_input('Insert night charge')
    customer_calls = st.sidebar.number_input('Insert no of customer_calls')
    
    #code for prediction (the result of prediction will return in this empty string)
    Churn = ''
    
    #creating button for prediction
    if st.button('Churn Result'):
        Churn = Churn_prediction([voice_plan_yes, voice_messages, intl_plan_yes, intl_mins, intl_calls, intl_charge, day_mins, day_calls, day_charge, eve_mins, eve_calls, eve_charge, night_mins, night_calls, night_charge, customer_calls])
    
    
    st.success('The output is {}'.format(Churn))
    
    
    
if __name__ == '__main__':
    main()



