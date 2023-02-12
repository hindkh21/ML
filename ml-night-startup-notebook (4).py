#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import read_csv
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as  pd 


# # loading the data

# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('Submission Sample.csv')


# In[4]:


train.head()


# In[5]:


submission.head()


# In[6]:


train.info()


# In[7]:


train.shape
train.columns


# In[8]:


train1= train.dropna(axis=0, how='any') 


# In[9]:


train1.shape


# In[10]:


train.describe()


# In[11]:


train.duplicated().unique()


# In[12]:


train.shop.unique()


# In[13]:


cols = ['start_date','end_date','format','device','iremoteid']

for col in cols:
    print(f"{col} has {train[col].unique()} values\n")
    


# In[14]:


train = train.drop(['device'],axis=1)
test = test.drop(['device'],axis=1)


# In[15]:


train.shape


# In[16]:


colObj=train.dtypes[train.dtypes=="object"].index.tolist()
for i in colObj : 
    print(train[i].value_counts())


# In[17]:


train = train.drop(['iremoteid','campaign_id','chain_id','shop'],axis=1)
test = test.drop(['iremoteid','campaign_id','chain_id','shop'],axis=1)

train.shape


# In[ ]:





# In[18]:


"train['shop'].value_counts()"



# In[19]:


train['height'].value_counts()


# In[20]:


train['width'].value_counts()


# In[21]:


train['budget'].value_counts()


# In[22]:


train['start_date']=pd.to_datetime(train['start_date'])
train['end_date']=pd.to_datetime(train['end_date'])


test['start_date']=pd.to_datetime(test['start_date'])
test['end_date']=pd.to_datetime(test['end_date'])


# In[23]:


train.info()


# In[24]:


train = train.drop(['start_date','end_date'],axis=1)

test = test.drop(['start_date','end_date'],axis=1)

train.shape


# In[25]:


train = pd.get_dummies(train, columns=['format'])
test = pd.get_dummies(test, columns=['format'])


train.info()


# # data processing

# In[26]:


#  this is the worst data processing ever 
# you should change this 
#train = train.drop(['start_date','end_date','format','device','iremoteid'],axis=1)
#test = test.drop(['start_date','end_date','format','device','iremoteid'],axis=1)


# # creating the X and y 

# In[27]:


y = train['budget']
X = train.drop(columns='budget')


# In[28]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[29]:


y_train


# # creating the model

# In[30]:


#from sklearn.linear_model import LinearRegression 

#model = LinearRegression()
#model.fit(X_train,y_train)
#model.score(X_test,y_test)


# In[31]:


pip install xgboost


# In[96]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000],
           'colsample_bytree': [0.3, 0.7]}
xgbr = xgb.XGBRegressor(seed = 20)
clf = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring='neg_mean_squared_error', 
                   verbose=1)
clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))


# # evaluaating the model 

# In[91]:





# In[97]:


# note that the lower the MSE the better 
from sklearn.metrics import mean_squared_error as mse 
y_preds = clf.predict(X_test)
rmse = mse(y_test,y_preds)**(1/2)
print(rmse)


# In[45]:


train.head()


# In[46]:


X.head()


# In[47]:


test.head()


# In[48]:


train[['height']].boxplot()


# In[ ]:





# In[ ]:





# In[49]:


train.info()


# # creating the submission

# In[93]:


preds = xg_reg.predict(test)


# In[51]:


submission['budget'] = preds


# In[101]:


# this code will generate a file that you should download 
submission.to_csv('FINAL2.csv',index=False)


# In[102]:


get_ipython().system('start .')


# # download this file 
# ![image.png](attachment:da575689-cba6-45c6-a27d-40666f0d50b2.png)

# # go back to the competition and submit the file there 

# ![image.png](attachment:6e894c79-fb15-4894-8deb-ff2d4aea6bc3.png)

# In[ ]:





# In[ ]:





# In[ ]:




