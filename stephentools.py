#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pydot
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# ## For standardization: 
# Depending on the task objetives. For example; for neural networks is recommended normalization Min max for activation functions. To avoid saturation Basheer & Najmeer (2000) recommend the range 0.1 and 0.9.

# # Prep Model

# In[15]:

def get_file():
    df=pd.read_csv(r'D3.csv')
    a=10.986409919681789
    b=171.91508511054587
    df['height']=df['height']*a+b       
    return df

def data_d3(): 
    df = get_file()
    df['height']=df.height.round()
    df['contacts_count']=df.contacts_count.round()
    #Height is Z-score Normalized  => need to change => reverse Z-score and convert to int32
    # contacts_count in assessment 1 have been imputed with MEAN => round it into int32
    mapping = {'yes':1, 'no':0, 'blank':np.nan}
    df['insurance']=df.insurance.map(mapping)
    df['insurance'].fillna(df['insurance'].mode()[0], inplace=True)
    secmap={'native':1,'immigrant':0, 'blank':np.nan}
    df['immigrant']=df.immigrant.map(secmap)
    df['immigrant'].fillna(df['immigrant'].mode()[0], inplace=True)
    #Convert data type: 
    convert_dict = {'contacts_count':int, 'height': int,'worried': int, 'immigrant':bool,'insurance':bool,'covid19_positive':bool,'covid19_symptoms':bool,'covid19_contact':bool,'asthma':bool,'kidney_disease':bool,'liver_disease':bool,'compromised_immune':bool,'heart_disease':bool,'lung_disease':bool,'diabetes':bool,'hiv_positive':bool,'hypertension':bool,'other_chronic':bool,'nursing_home':bool,'health_worker':bool}
    df = df.astype(convert_dict) 

    df = pd.get_dummies(df)
    y = df['covid19_positive']
    X = df.drop(['covid19_positive'], axis=1)
    
    
    # setting random state
    rs = 42

    X_mat = X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.29, stratify=y, random_state=rs)

    return df,X,y,X_train, X_test, y_train, y_test

# In[16]:


# For Regression: 
def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i], ':', importances[i])


# In[17]:


# Decision Tree 
def visualize_decision_tree(dm_model, feature_names, save_name):
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph[0].write_png(save_name) # saved in the following file



# In[5]:





# In[ ]:





# In[6]:





# In[ ]:





# In[8]:





# In[9]:





# In[ ]:





# In[10]:





# In[ ]:





# In[12]:





# In[13]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:





# In[18]:





# In[ ]:




