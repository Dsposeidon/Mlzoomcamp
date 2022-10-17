#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('AER_credit_card_data.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

card_values = {
    "yes": 1,
    "no": 0
}
df["card"] = df.card.map(card_values)


# ## Spliting the model

# In[3]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[4]:


numerical = ["reports", "age", "income", "share", "expenditure", "dependents", "months", "majorcards", "active"]
categorical = ["owner", "selfemp"]


# In[5]:


features = ['reports', 'share', 'expenditure', 'owner']

def train(df_train, y_train, C=1.0):
    dicts = df_train[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[7]:


C = 1.0
n_splits = 5


# In[11]:


scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.card.values
    y_val = df_val.card.values

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[12]:


scores


# In[14]:


dv, model = train(df_full_train, df_full_train.card.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.card.values
auc = roc_auc_score(y_test, y_pred)
auc


# ### Load the Model

# In[18]:


import pickle


# In[19]:


output_file = f'model_C={C}.bin'
output_file


# In[20]:


f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# In[21]:


dv, model


# In[46]:


customer = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}


# In[47]:


X = dv.transform([customer])


# In[48]:


model.predict_proba(X)[0,1]


# In[ ]:




