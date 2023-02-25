#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[3]:


import streamlit as st 
st.title('Model Deployment: Logistic Regression')


# In[4]:


st.sidebar.header('User Input Parameters')


# In[5]:


def user_input_features():
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
    
    
    
    
    data = {'voice_plan':voice_plan_yes,
            'Voice_messages':voice_messages,
            'intl_plan':intl_plan_yes,
            'intl_mins':intl_mins,
            'intl_calls':intl_calls,
            'intl_charge':intl_charge,
            'day_mins':day_mins,
            'day_calls':day_calls,
            'day_charge':day_charge,
            'eve_mins':eve_mins,
            'eve_calls':eve_calls,
            'eve_charge':eve_charge,
            'night_mins':night_mins,
            'night_calls':night_calls,
            'night_charge':night_charge,
            'customer_calls':customer_calls}
    
    features = pd.DataFrame(data,index = [0])
    return features 


# In[6]:


churn = user_input_features()
st.subheader('User Input parameters')
st.write(churn)


# In[7]:


df = pd.read_csv("E:/Userfiles/Download/Churn.csv")
df


# In[8]:


df.info()


# In[9]:


df1 = df.copy()
df1["day.charge"] = df1["day.charge"].astype("float64")
df1["eve.mins"]= df1["eve.mins"].astype("float64")


# In[10]:


df1.isnull().sum()


# In[11]:


df1.dropna(inplace = True)


# In[12]:


df1.shape


# In[13]:


df1.drop(df1.columns[[0, 1, 2, 3]], axis=1, inplace=True)
df1


# In[14]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df1['churn']=LE.fit_transform(df1['churn'])


# In[15]:


df1=pd.get_dummies(df1,columns=['voice.plan','intl.plan'], drop_first=True)
df1


# In[16]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[17]:


df_norm = norm_func(df1)


# In[18]:


X = df_norm.drop(['churn'], axis=1)
y = df_norm['churn']


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)


# In[21]:


Xtrain.shape , Xtest.shape , ytrain.shape , ytest.shape 


# In[22]:


from imblearn.over_sampling import RandomOverSampler
OS=RandomOverSampler(sampling_strategy=0.90)
Xtrain_os,ytrain_os=OS.fit_resample(Xtrain,ytrain)


# In[23]:


from collections import Counter
print("The number of classes before fit {}".format(Counter(ytrain)))
print("The number of classes after fit {}".format(Counter(ytrain_os)))


# In[24]:


Xtrain_os.head()


# # Random Forest

# In[25]:


from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier()
Classifier.fit(Xtrain_os, ytrain_os)


# In[26]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
y_pred = Classifier.predict(Xtest)
print(confusion_matrix(ytest,y_pred))
print(accuracy_score(ytest,y_pred))
print(classification_report(ytest,y_pred))


# In[27]:


prediction_proba = Classifier.predict_proba(Xtest)
prediction_proba


# In[28]:


st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[29]:


import pickle


# In[30]:


filename = 'Churn_predicted.pkl'
pickle.dump(Classifier,open(filename,'wb'))


# In[31]:


loaded_model = pickle.load(open('Churn_predicted.pkl','rb'))


# In[ ]:





# In[ ]:





# In[ ]:




