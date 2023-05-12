#!/usr/bin/env python
# coding: utf-8

# <div class="markdown-google-sans">
# 
# ## Data science
# </div>
# 
# With Colab you can harness the full power of popular Python libraries to analyze and visualize data. The code cell below uses **numpy** to generate some random data, and uses **matplotlib** to visualize it. To edit the code, just click the cell and start editing.

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle


# In[2]:


data = pd.read_csv('your.csv') 


# In[3]:


data.describe()


# In[4]:


data.dropna(inplace=True)


# In[5]:


data


# In[6]:


data.drop(['avg_quiz2','avg_pract','avg_final'], axis=1, inplace=True)


# In[7]:


data


# In[15]:


X = data[['avg_quiz1', 'avg_mid']]  # Features (inputs)
y = data['totalGPA']  # Target variable (output)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
model = LinearRegression()  # Initialize the linear regression model
model.fit(X_train, y_train)  # Train the model


# In[16]:


# Model evaluation
y_pred = model.predict(X_test)  # Make predictions on the test set
y_pred


# In[17]:


ms = model.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
r2 = r2_score(y_test, y_pred)  # Calculate R-squared
print("Mean Squared Error:", mse)
print("model-score:", ms)
print("R-squared:", r2)


# In[ ]:




