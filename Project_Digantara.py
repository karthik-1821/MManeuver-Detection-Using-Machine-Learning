#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime


# In[2]:


df = pd.read_csv("SMA_data.csv")
df


# In[3]:


df = pd.read_csv("SMA_data.csv")
print(df.columns)


# In[4]:


df['Datetime'] = pd.to_datetime(df['Datetime'])
df['SMA_change'] = df['SMA'].diff()
df.dropna(inplace=True)


# In[5]:


maneuver_dates = ['2018-05-03', '2018-10-11', '2019-03-27', '2019-05-17', '2019-09-11', '2019-11-01']
df['Maneuver'] = df['Datetime'].apply(lambda x: 1 if x.strftime('%Y-%m-%d') in maneuver_dates else 0)


# In[6]:


X = df[['SMA', 'SMA_change']]
y = df['Maneuver']


# In[7]:


print(X)
print(y)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[10]:


plt.figure(figsize=(10, 6))
plt.plot(df['Datetime'], df['SMA'], label='SMA', color='blue')
plt.scatter(df[df['Maneuver'] == 1]['Datetime'], df[df['Maneuver'] == 1]['SMA'], 
            color='red', label='Maneuver Detected')
plt.xlabel('Datetime')
plt.ylabel('SMA')
plt.legend()
plt.show()


# In[ ]:




