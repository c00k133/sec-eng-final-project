# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
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
# %pip install modin[all] pandas==1.2.3

# %%
'''
try:
    import modin.pandas as pd
except ModuleNotFoundError:
    import pandas as pd
'''
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os

from zipfile import ZipFile, Path
from io import StringIO

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# %% [markdown]
# ---
#
# ### Styling

# %%
sns.set(style='whitegrid')

HAM_COLOR = '#0000ff99'
SPAM_COLOR = '#ff000099'


def save_figure(filename, figure_dir='illustrations', dpi=100, *args, **kwargs):
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir, exist_ok=True)
    fullpath = os.path.join(figure_dir, filename)
    plt.savefig(fullpath, dpi=dpi, *args, **kwargs)


# %% [markdown]
# ---

# %%
column_names = [
    'IsSpam',
    'Message',
]

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
DATA_ZIP_FILE = 'archive.zip'
if not os.path.exists(DATA_ZIP_FILE):
    req = requests.get(DATASET_URL, DATA_ZIP_FILE)
    with open(DATA_ZIP_FILE, 'wb+') as fd:
        fd.write(req.content)

raw_data = Path(DATA_ZIP_FILE, at='SMSSpamCollection')

raw_df = pd.read_csv(
    raw_data.open(),
    encoding='latin-1', # Inferred from other Kaggle notebooks
    header=0, # Required when using explicit names
    delimiter='\t',
    names=column_names,
    skipinitialspace=False,
)

raw_df.describe()

# %% [markdown]
# Noticed a bunch of extra column values, probably unescaped commas in the messages.
#
# This was only the case for data downloaded from Kaggle, the data from the paper site did not have this issue.

# %%
'''
extra_columns = [
    'Unknown1',
    'Unknown2',
    'Unknown3',
]

long_messages = raw_df[
    raw_df.unknown1.notna() | raw_df.unknown2.notna() | raw_df.unknown3.notna()
]
long_messages.describe()

message_names = column_names[1:]
concatenated = ','.join(long_messages.loc[281][message_names])
print(concatenated)

def concatenate_extras(row):
    non_empty_parts = row[message_names].dropna()
    return ','.join(non_empty_parts)

concatenated_messages = long_messages.apply(concatenate_extras, axis='columns')
raw_df.loc[concatenated_messages.index, 'message'] = concatenated_messages

extra_columns = message_names[1:]
sms = raw_df.drop(columns=extra_columns)
'''

# %% [markdown]
# Much better...

# %%
sms = raw_df
sms.loc[sms['IsSpam'] == 'spam', 'SpamValue'] = 1
sms.loc[sms['IsSpam'] == 'ham', 'SpamValue'] = 0

# %%
spam_or_ham_count = sms.IsSpam.value_counts()
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

save_figure('spam_or_ham_pie.png')

# %%
nltk.download('stopwords')

# %%
slang_stopwords = ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
singlish_stopwords = [
    'lah', 'lor', 'shiok', 'bojio', 'like',
    'hor', 'sian', 'walau', 'eh', 'damn',
    'ya', 'wah', 'bb', 'leh', 'lar'
]
considered_stopwords = {
    *stopwords.words('english'),
    *slang_stopwords,
    #*singlish_stopwords,
}


# %%
def create_wordcloud(text, stopwords=considered_stopwords, *args, **kwargs):
    wordcloud = WordCloud(
        stopwords=stopwords,
        background_color='white',
        width=800,
        height=800,
        *args, **kwargs
    )
    wordcloud_figure = wordcloud.generate(text)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud_figure)
    plt.axis('off')


# %%
ham_text = ' '.join(sms[sms.IsSpam == 'ham']['Message'].tolist())
create_wordcloud(ham_text, colormap='viridis')

save_figure('ham_wordcloud.png')

# %%
spam_text = ' '.join(sms[sms.IsSpam == 'spam']['Message'].tolist())
create_wordcloud(spam_text, colormap='inferno')

save_figure('spam_wordcloud.png')

# %%
sms['MessageLength'] = sms.Message.apply(len)
sms[['MessageLength']].value_counts()

# %%
sms[sms.IsSpam == 'ham'].MessageLength.plot.hist(
    bins=35,
    figsize=(12, 6),
    
    legend=True,
    color=HAM_COLOR,
    label='Ham messages',
)
sms[sms.IsSpam == 'spam'].MessageLength.plot.hist(
    legend=True,
    color=SPAM_COLOR,
    label='Spam messages',
)

plt.xlabel('Message length')

save_figure('message_length_histogram.png')

# %%
sms.groupby('IsSpam')[['MessageLength']].describe()

# %% [markdown]
# ---
#
# ## Analytics

# %%
'''
try:
    import modin.pandas as pd
except ModuleNotFoundError:
    import pandas as pd
'''
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import os
import string

from functools import partial, reduce

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords

# %%
nltk.download('punkt')
nltk.download('stopwords')


# %% [markdown]
# Remove capitalization unless the word is all upper case.
# Lowercase all caps words might remove messge sentiment.
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
#
# Data science article regarding Singaporean text mining:
# https://towardsdatascience.com/topic-modeling-singapores-subreddit-comments-with-natural-language-processing-336d15da3ff4

# %%
def compose(*fns):
    return partial(reduce, lambda v, fn: fn(v), fns)

def filter_punctation(message):
    return ''.join(
        char
        for char in message
        if char not in string.punctuation
    )

def tokenize(message):
    return nltk.word_tokenize(message)

def lower_case_unless_all_caps(tokens):
    return (
        token.lower()
        for token in tokens
        if token.upper() != token
    )

def prune_stopwords(message_contents):
    return (
        word
        for word in message_contents
        if word.lower() not in considered_stopwords
    )

message_parsing_pipe = compose(
    #filter_punctation,
    tokenize,
    #lower_case_unless_all_caps,
    prune_stopwords,
    list,
)

sms["Tokens"] = sms["Message"].apply(message_parsing_pipe)

# %% [markdown]
# https://franekjemiolo.pl/flattening-lists-pandas/

# %%
flattened_tokens = pd.DataFrame(
    [
        (index, value, 1)
        for (index, values) in sms['Tokens'].iteritems()
        for value in values
    ],
    columns=['index', 'Token', 'Count']
).set_index('index')

flattened = sms[['IsSpam']].join(flattened_tokens)
count = flattened.pivot_table(
    index='Token',
    columns='IsSpam',
    values='Count',
    aggfunc=np.sum,
    fill_value=0
)
token_data = count.reset_index().rename(columns={'ham': 'nHam', 'spam': 'nSpam'})

# %%
token_data["Probability"] = token_data["nHam"] / (token_data["nSpam"] + token_data["nHam"])
token_data.sort_values(by=["Probability"], ascending=False).head(10)

# %%
identifiers = sms.copy()
identifiers = sms.rename(columns={"Message": "OriginalMessage"}, errors="raise")

# %%
for item in token_data["Token"].unique():
    identifiers[item] = 0

# %% [markdown]
# The cell below could really benefit from some optimization...
# If using `.pivot_table(...)` would be possible we could get insane speed bumps!

# %%
for item in token_data["Token"].unique():
    identifiers.loc[identifiers["OriginalMessage"].str.contains(item, regex=False), item] = 1

# %%
is_spam = pd.to_numeric(identifiers["SpamValue"])
identfiers_matrix = identifiers.drop(["OriginalMessage", "Tokens", "IsSpam", "SpamValue", "MessageLength"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(identfiers_matrix, is_spam, test_size=0.2, random_state=0)

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
success = sum(y_test.array == predictions) / X_test.shape[0]
success

# %%
identfiers_matrix
