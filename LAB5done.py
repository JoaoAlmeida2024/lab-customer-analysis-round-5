#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


df = pd.read_csv(r"C:\Users\joaof\ironhack\labs\lab-customer-analysis-round-2\files_for_lab\csv_files\marketing_customer_analysis.csv")


# In[5]:


df = pd.read_csv(r"C:\Users\joaof\ironhack\labs\lab-customer-analysis-round-4\files_for_lab\csv_files\marketing_customer_analysis.csv")


# In[6]:


df


# In[7]:


df.shape


# In[8]:


df.head(5)


# In[9]:


cols = []


# In[10]:


for i in range(len(df.columns)):
    cols.append(df.columns[i].lower().replace(' ', '_'))


# In[11]:


df.columns = cols


# In[12]:


df


# In[13]:


numeric_columns = df.select_dtypes(include=['int', 'float'])


# In[14]:


numeric_columns


# In[15]:


categorical_columns = df.select_dtypes(include= ['object'])


# In[16]:


categorical_columns


# In[17]:


import seaborn as sns


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


sns.histplot(data=numeric_columns, x = 'income')
plt.ylabel('frequency')
plt.xlabel('income')
plt.show()


# In[20]:


sns.histplot(data=numeric_columns, x = 'monthly_premium_auto')
plt.ylabel('frequency')
plt.xlabel('monltlhy_premium_auto')
plt.show()


# In[21]:


sns.histplot(data=numeric_columns, x = 'income')
plt.ylabel('frequency')
plt.xlabel('monltlhy_premium_auto')
plt.show()


# In[22]:


sns.displot(data=numeric_columns, x = 'months_since_last_claim')
plt.ylabel('frequency')
plt.xlabel('Months Since Last Claim')
plt.show()


# In[23]:


sns.displot(data=numeric_columns, x = 'number_of_policies')
plt.ylabel('frequency')
plt.xlabel('Number of open complaints')
plt.show()


# In[24]:


sns.displot(data=numeric_columns, x = 'number_of_open_complaints')
plt.ylabel('frequency')
plt.xlabel('Number of open complaints')
plt.show()


# In[25]:


correlation_matrix =  numeric_columns.corr()


# In[26]:


correlation_matrix


# In[27]:


sns.heatmap(correlation_matrix, annot=True)
plt.show()
mask = np.zeros_like(correlation_matrix)
mask[np.triu_indices_from(mask)] = True 
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(correlation_matrix, mask=mask, annot=True)
plt.show()


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


df_train, df_test = train_test_split(df, test_size=0.2, random_state=100)
print(df_train.shape, df_test.shape)


# In[31]:


#X-Y SPLIT
X = df.drop('customer_lifetime_value', axis=1)
Y= df['customer_lifetime_value']


# In[32]:


print("Features (X):")
print(X)
print("\nTarget (Y):")
print(Y)


# In[ ]:




