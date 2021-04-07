# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SMS Spam
#
# ---

# %% [markdown]
# > https://www.kaggle.com/uciml/sms-spam-collection-dataset

# %% [markdown]
# ---
#
# ## EDA

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ---
#
# ### Styling

# %%
sns.set(style='whitegrid')

HAM_COLOR = '#0000ff99'
SPAM_COLOR = '#ff000099'

# %% [markdown]
# ---

# %%
column_names = [
    'spam_or_ham',
    'message',
    'unknown1',
    'unknown2',
    'unknown3',
]

raw_df = pd.read_csv(
    #'./archive.zip',
    #compression='zip',
    'spam.csv',
    encoding='latin-1', # Inferred from other Kaggle notebooks
    header=0, # Required when using explicit names
    names=column_names,
    skipinitialspace=False,
)

raw_df.describe()

# %% [markdown]
# Noticed a bunch of extra column values, probably unescaped commas in the messages.

# %%
long_messages = raw_df[
    raw_df.unknown1.notna() | raw_df.unknown2.notna() | raw_df.unknown3.notna()
]
long_messages.describe()

# %% [markdown]
# Certainly looks like it...

# %%
message_names = column_names[1:]
concatenated = ','.join(long_messages.loc[281][message_names])
print(concatenated)


# %% [markdown]
# Let's concatenate them!

# %%
def concatenate_extras(row):
    non_empty_parts = row[message_names].dropna()
    return ','.join(non_empty_parts)

concatenated_messages = long_messages.apply(concatenate_extras, axis='columns')
raw_df.loc[concatenated_messages.index, 'message'] = concatenated_messages

extra_columns = message_names[1:]
df = raw_df.drop(columns=extra_columns)

# %% [markdown]
# Much better...

# %%
df.describe()

# %%
spam_or_ham_count = df.spam_or_ham.value_counts()
spam_or_ham_count.plot.pie(
    legend=False,
    ylabel='',
    autopct='%.1f%%',
    explode=[0.00, 0.05],
    startangle=0,
    labels=['Ham', 'Spam'],
    figsize=(5, 5),
    title='Ham to Spam ratio',
    colors=[HAM_COLOR, SPAM_COLOR],
)

# %%
df['message_length'] = df.message.apply(len)
df[['message_length']].value_counts()

# %%
df[df.spam_or_ham == 'ham'].message_length.plot.hist(
    bins=35,
    figsize=(12, 6),
    
    legend=True,
    color=HAM_COLOR,
    label='Ham messages',
)
df[df.spam_or_ham == 'spam'].message_length.plot.hist(
    legend=True,
    color=SPAM_COLOR,
    label='Spam messages',
)

plt.xlabel('Message length')

# %% [markdown]
# ---
#
# ## Analytics

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords

# %%
NLTK_DATA_DIR = './nltk_data'
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# %%
nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('stopwords', download_dir=NLTK_DATA_DIR)

# %%
#data = pd.read_csv("spam.csv", encoding='ISO-8859-1') 
#data = data[['v1', 'v2']]
#data = data.rename(columns={"v1": "IsSpam", "v2": "Message"}, errors="raise")
data = df.rename(columns={'spam_or_ham': 'IsSpam', 'message': 'Message'}, errors='raise')
data.loc[data["IsSpam"] == "spam", "IsSpam"] = 1
data.loc[data["IsSpam"] == "ham", "IsSpam"] = 0


# %% [markdown]
# Remove capitalization unless the word is all upper case.
#
# http://sentiment.christopherpotts.net/tokenizing.html#capitalization
#
# Although it might make sense to lower case everything, depending on how much sentiment analysis we want to include
#
# https://towardsdatascience.com/effectively-pre-processing-the-text-data-part-1-text-cleaning-9ecae119cb3e
#
# _NOTE_:
# Didn't seem to have a difference...
# The random tree walk performed worst when we used lower-not-for-caps casing.
# The all lower was worst for Naïve Bayes.

# %%
def lower_case_unless_all_caps(word):
    if word.upper() == word:
        return word
    return word.lower()

def tokenize(message):
    tokenized = nltk.word_tokenize(message)
    return [
        #lower_case_unless_all_caps(word)
        #word.lower()
        word
        for word in tokenized
    ]

data["Tokens"] = data["Message"].apply(tokenize)

# %%
stop = stopwords.words('english')
data["Tokens"] = data["Tokens"].apply(lambda x: [item for item in x if item not in stop])
# tokenList = list(set([item for sublist in data["Tokens"].tolist() for item in sublist]))
# print(len(tokenList))

# %%
tokenList = list(set([item for sublist in data["Tokens"].tolist() for item in sublist]))
tokenData = pd.DataFrame(tokenList, columns = ["Words"])
tokenData["nSpam"] = 0
tokenData["nHam"] = 0

# %%
print(data.shape)
for index, row in data[["IsSpam", "Tokens"]].iterrows():
    columnName = "nSpam" if row.IsSpam == 1 else "nHam"
    for token in row["Tokens"]:
        tokenData.loc[tokenData["Words"] == token, columnName] += 1

# %%
tokenData

# %%
# tokenData["Difference"] = tokenData["nSpam"] - tokenData["nHam"]
# tokenData.sort_values(by=["Difference"], ascending=False).head(10)

tokenData["Probability"] = (tokenData["nSpam"] - tokenData["nHam"]) / (tokenData["nSpam"] + tokenData["nHam"])
tokenData.sort_values(by=["Probability"], ascending=False).head(10)

# %%
identifiers = data.copy()
identifiers = data.rename(columns={"Message": "Original_Message"}, errors="raise")

# %%
for item in tokenData["Words"].tolist():
    identifiers[item] = 0

# %%
for item in tokenData["Words"].tolist():
    identifiers.loc[identifiers["Original_Message"].str.contains(item, regex=False), item] = 1

# %%
isSpam = pd.to_numeric(identifiers["IsSpam"])
identfiers_matrix = identifiers.drop(["Original_Message", "Tokens", "IsSpam"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(identfiers_matrix, isSpam, test_size=0.2, random_state=0)

# %%
gnb = GaussianNB()
model = gnb.fit(X_train, y_train)
result = model.predict(X_test)
success = sum(y_test.array == result) / X_test.shape[0]
success

# %%
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
success = sum(y_test.array == predictions)/X_test.shape[0]
success

# %%
identfiers_matrix
