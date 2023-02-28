# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:29:05 2023

@author: Jeevika
"""

import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('https://github.com/SundaramAyyappan/deploy/blob/main/Churn_prediction.sav','rb'))

#Creating a function for prediction

def Churn_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person will not be churned'
    else:
        return 'The person will be churned'
    


def main():
    
    st.title('Churn prediction Web App')
    st.sidebar.header('User Input Parameters')
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
        churn = Churn_prediction([voice_plan_yes, voice_messages, intl_plan_yes, intl_mins, intl_calls, intl_charge, day_mins, day_calls, day_charge, eve_mins, eve_calls, eve_charge, night_mins, night_calls, night_charge, customer_calls])
    
    st.success(Churn)
    
    
    if __name__ == '__main__':
        main()





