#!/usr/bin/env python
# coding: utf-8

# # Credit Card Default Prediction

# The data set consists of 2000 samples from each of two categories. Five variables are
# 
# 1. Income
# 2. Age
# 3. Loan
# 4. Loan to Income (engineered feature)
# 5. Default

# In[1]:


# Step 1 : import library
import pandas as pd


# In[2]:


# Step 2 : import data
default = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv')


# In[3]:


default.head()


# In[4]:


default.info()


# In[5]:


default.describe()


# In[6]:


# Count of each category
default['Default'].value_counts()


# In[7]:


# Step 3 : define target (y) and features (X)
     
default.columns
     


# In[8]:


y = default['Default']


# In[9]:


X = default.drop(['Default'],axis=1)


# In[10]:


# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)


# In[11]:


# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[12]:


# Step 5 : select model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[13]:


# Step 6 : train or fit model
model.fit(X_train,y_train)


# In[15]:


model.intercept_


# In[16]:


model.coef_


# In[17]:


# Step 7 : predict model
y_pred = model.predict(X_test)


# In[18]:


y_pred


# In[19]:


# Step 8 : model accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[20]:


confusion_matrix(y_test,y_pred)


# In[21]:


accuracy_score(y_test,y_pred)


# In[22]:


print(classification_report(y_test,y_pred))


# In[ ]:




