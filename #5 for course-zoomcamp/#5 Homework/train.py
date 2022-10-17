#!/usr/bin/env python
# coding: utf-8

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
import pickle

## parameters

C = 1.0
n_splits = 5

output_file = f'model_C={C}.bin'


# Data Preparation 

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


# ### Spliting the model

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)



numerical = ["reports", "age", "income", "share", "expenditure", "dependents", "months", "majorcards", "active"]
categorical = ["owner", "selfemp"]



features = ['reports', 'share', 'expenditure', 'owner']


# Training model

def train(df_train, y_train, C=1.0):
    dicts = df_train[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[features].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred




# validation

print(f'doing validation with C={C}')


scores = []
fold = 0

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

    print('auc on fold {fold} is {auc}')
    fold = fold + 1


print(f'auc={auc}')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))





dv, model = train(df_full_train, df_full_train.card.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.card.values
auc = roc_auc_score(y_test, y_pred)
auc

print(f'auc={auc}')

# ### Save the Model


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)



print(f'the model is saved to {output_file} ')








