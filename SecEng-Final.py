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
# ### Analytical SMS Spam Detection
#
# Group members:
# - Alazar Abebaw - alazar.abebaw@aalto.fi - 915739
# - Anmol Sinha - anmol.sinha@aalto.fi - 885911
# - Ujjwol Dandekhya - ujjwol.dandekhya@aalto.fi - 899237
# - Axel Neergaard - axel.neergaard@aalto.fi - 529840
#
# This project on GitHub:
# > https://github.com/c00k133/sec-eng-final-project

# %% [markdown]
# ---
#
# ### Acknowledgments
#
# The dataset used for this project is from a machine learning repository provided by the University of California, Irvine (UCI):
# > https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
#
# Below is a Kaggle dataset based on the UCI dataset:
# > https://www.kaggle.com/uciml/sms-spam-collection-dataset
#
# #### References
#
# This is not an exhaustive list of sources, but should provide an overview of consulted material:
# - https://towardsdatascience.com/effectively-pre-processing-the-text-data-part-1-text-cleaning-9ecae119cb3e
# - https://towardsdatascience.com/topic-modeling-singapores-subreddit-comments-with-natural-language-processing-336d15da3ff4
# - https://link.springer.com/chapter/10.1007%2F978-3-642-31178-9_12
# - http://sentiment.christopherpotts.net/tokenizing.html#capitalization
# - https://towardsdatascience.com/effectively-pre-processing-the-text-data-part-1-text-cleaning-9ecae119cb3e
# - https://towardsdatascience.com/topic-modeling-singapores-subreddit-comments-with-natural-language-processing-336d15da3ff4

# %% [markdown]
# ---
#
# ## Table of contents
#
# - Common
#     - Common setup for all sections.
# - Exploratory Data Analysis (EDA)
#     - EDA of the dataset
# - Analysis
#     - Analysis containing classification models for spam detection
# - Demo
#     - A short demo that can be used to evaluate short phrases

# %% [markdown]
# ---
#
# ## Common
#
# This section of the notebook includes common setup for all other parts of the notebook.
# This includes:
# - styling;
# - utility tooling;
# - data extraction and setup;
# - and stopword creation.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from textwrap import dedent

import nltk
from nltk.corpus import stopwords

# %% [markdown]
# ### Styling & Utilities

# %% [markdown]
# Let's create some uniform styling for the project:

# %%
sns.set(style='whitegrid')

HAM_COLORS = {
    'primary': '#0000ff99',
    'colormap': 'viridis',
}
SPAM_COLORS = {
    'primary': '#ff000099',
    'colormap': 'inferno',
}

CONFUSION_MATRIX = {
    'colormap': 'Reds',
}

PIE_CHART_STYLE = {
    'autopct': '%.1f%%',
    'explode': [0.00, 0.05],
    'startangle': 0,
    'colors': [HAM_COLORS['primary'], SPAM_COLORS['primary']],
}

# %% [markdown]
# Set the following flag to `True` in case you want to save all illustrations:

# %%
SAVE_ILLUSTRATIONS = True

# %% [markdown]
# Enable this flag to optimize the illustrations for the paper:

# %%
OPTIMIZE_ILLUSTRATIONS = True


# %% [markdown]
# A utility function for saving illustrations and graphs:

# %%
def save_figure(filename, figure_dir='illustrations', dpi=100, *args, **kwargs):
    if not SAVE_ILLUSTRATIONS:
        return
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir, exist_ok=True)

    fullpath = os.path.join(figure_dir, filename)
    plt.savefig(
        fullpath,
        dpi=dpi,
        bbox_inches=None if not OPTIMIZE_ILLUSTRATIONS else 'tight',
        *args, **kwargs
    )


# %% [markdown]
# Utility functions for various metrics based on confusion matrices:

# %%
def calculate_accuracy(cf):
    return np.trace(cf) / float(np.sum(cf))

def calculate_precision(cf):
    true_positive = cf[1, 1]
    false_pos_true_pos = cf[:, 1]
    return true_positive / sum(false_pos_true_pos)

def calculate_recall(cf):
    true_positive = cf[1, 1]
    false_neg_true_pos = cf[1, :]
    return true_positive / sum(false_neg_true_pos)

def calculate_f1_score(cf):
    precision = calculate_precision(cf)
    recall = calculate_recall(cf)
    return 2 * precision * recall / (precision + recall)


# %% [markdown]
# Below we have a function for drawing confusion matrices:

# %%
def paint_confusion_matrix(cf,
                           cbar=False,
                           xyticks=True,
                           figsize=None,
                           cmap=CONFUSION_MATRIX['colormap'],
                           title=None,
                           fig_filename=None):
    group_labels = ['True Neg\n', 'False Pos\n', 'False Neg\n', 'True Pos\n']
    group_counts = ['{0:0.0f}\n'.format(value) for value in cf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf.flatten() / np.sum(cf)]

    box_labels = [f'{v1}{v2}{v3}'.strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    accuracy = calculate_accuracy(cf)
    precision = calculate_precision(cf)
    recall = calculate_recall(cf)
    f1_score  = calculate_f1_score(cf)
    stats_text = '''\n
        Accuracy={:0.3f}
        Precision={:0.3f}
        Recall={:0.3f}
        F1 Score={:0.3f}
    '''.format(accuracy, precision, recall, f1_score)

    if not figsize:
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        categories = False

    plt.figure(figsize=figsize)
    categories = ['Spam', 'Ham']
    ax = sns.heatmap(
        cf,
        annot=box_labels,
        fmt='',
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories
    )

    plt.ylabel('True label')
    plt.xlabel('Predicted label' + dedent(stats_text))
    
    if title:
        ax.set_title(title)
        
    if fig_filename:
        save_figure(f'{fig_filename}.png')


# %% [markdown]
# ### Data extraction

# %% [markdown]
# We've automated the dataset download for your convenience.
# This is safe to re-run, as it checks whether the dataset already exists.
#
# The dataset can be found in the original UCI repository below:
# > https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

# %%
import requests
from zipfile import ZipFile, Path
from io import StringIO

# %%
DATA_ZIP_FILE = 'archive.zip'

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
if not os.path.exists(DATA_ZIP_FILE):
    req = requests.get(DATASET_URL, DATA_ZIP_FILE)
    with open(DATA_ZIP_FILE, 'wb+') as fd:
        fd.write(req.content)

# %% [markdown]
# We know from earlier analysis (discluded from this notebook) that there are two columns in the data.
#
# _NOTE_: peculiarly naming the "SMS message" column to `Message` would skew the results slightly as messages contained this word. Thus we've explitly named this column `Original_Message`.

# %%
column_names = [
    'SpamLabel',
    'Original_Message',
]

raw_data = Path(DATA_ZIP_FILE, at='SMSSpamCollection')

sms = pd.read_csv(
    raw_data.open(),
    encoding='latin-1', # Inferred from other Kaggle notebooks
    header=0, # Required when using explicit names
    delimiter='\t',
    names=column_names,
    skipinitialspace=False,
)

# %% [markdown]
# ### Stopwords
#
# Stopwords are common words that are omitted from datasets before natural language processing.
# Such words should not add to the sentiment to a message, and should be safe to remove from a sentence while still keeping the meaning intact.
# There is no standard on what constitutes a stopword.
# Thus, we are using stopwords provided by the Natural Language Toolkit (NLTK) library for Python, along with some slang commonly used in SMS messages, and a list of Singaporean specific slang (as a large portion of the messages originate from Singapore).
#
# The slang words were used in another project, while the Singaporean word list was built with the help from a native Singaporean.

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
    *singlish_stopwords,
}

# %% [markdown]
# ---
#
# ## Exploratory Data Analysis (EDA)
#
# We approached the EDA by scavenging the internet for various tutorials and nitbits, along with applying our own prior knowledge.
#
# As context for the EDA we knew we would apply some natural language processing in our analysis section.
# Hence the features and general analysis we performed kept any differences between spam or ham in mind.
#
# _A note on the terminology:_ when talking about spam messages one type of nomenclature uses "spam" to refer to spam messages, while "ham" for non-spam, legitimate messages.

# %%
from functools import reduce
from wordcloud import WordCloud

# %% [markdown]
# Let us first get an overview of the dataframe:

# %%
sms.head()

# %%
sms.describe()

# %% [markdown]
# As we can see, the dataframe only consists of two columns.
# The spam labelling column uses string labels, while many NLP applications use integers.
# Let's create a new column for NLP purposes:

# %%
sms['SpamValue'] = sms.SpamLabel.map({'spam': 1, 'ham': 0})

# %% [markdown]
# ### Spam vs. Ham ratio
#
# Before exploring the dataset heavily, it makes sense to get an overview of the ratio of ham versus spam messages.
# For this purpose, let's create a pie chart to show the relative ratio:

# %%
spam_or_ham_count = sms.SpamLabel.value_counts()
spam_or_ham_count.plot.pie(
    legend=False,
    ylabel='',
    labels=['Ham', 'Spam'],
    title='Ham to Spam ratio' if not (1.29, 0.9) else None,
    figsize=(6, 6) if not OPTIMIZE_ILLUSTRATIONS else (3.5, 3.5),
    **PIE_CHART_STYLE
)

save_figure('spam_or_ham_pie.png')


# %% [markdown]
# ### Word frequency
#
# Next we want to get an overview of the words used in ham and spam messages respectively.
# This overview might allow us to reason about patterns used in respective message types.
# Intuitively, and perhaps from experience, one might think that spam messages include words used in "call of action" phrase (i.e. phrases attempting to evoke some actionable reaction from a user, e.g. visiting a webpage).
#
# For this purpose we decided to use a wordcloud.
# A wordcloud is a great human-friendly image that highlights the most commonly used words in a text.
# This should allow us to see commonly used words in an instant.

# %%
def create_wordcloud(text, stopwords=considered_stopwords, *args, **kwargs):
    wordcloud = WordCloud(
        stopwords=stopwords,
        background_color='white',
        width=800 if not OPTIMIZE_ILLUSTRATIONS else 1600,
        height=800,
        margin=0,
        *args, **kwargs
    )
    wordcloud_figure = wordcloud.generate(text)
    
    plt.figure(
        figsize=(12, 12) if not OPTIMIZE_ILLUSTRATIONS else (12, 6)
    )
    plt.imshow(wordcloud_figure)
    plt.axis('off')


# %% [markdown]
# First we create a wordcloud for ham messages:

# %%
ham_text = ' '.join(sms[sms.SpamLabel == 'ham']['Original_Message'].tolist())
create_wordcloud(ham_text, colormap=HAM_COLORS['colormap'])

save_figure('ham_wordcloud.png')

# %% [markdown]
# Then we create a wordcloud for spam messages:

# %%
spam_text = ' '.join(sms[sms.SpamLabel == 'spam']['Original_Message'].tolist())
create_wordcloud(spam_text, colormap=SPAM_COLORS['colormap'])

save_figure('spam_wordcloud.png')

# %% [markdown]
# As we can notice, the spam messages seem to include words that are trying to evoke action from a used, such as the words `FREE` and `Urgent` in all-caps.
#
# Ham messages on the other hand seem to have include benign words used in everyday speech.

# %% [markdown]
# ### Message length
#
# Another possibly interesting feature that we may extract from the data is related to message length.
#
# Intuitively, legitimate SMS messages might skew on the shorter side.
# This is to save time and costs on the sender side as humans are able to parse even badly malformed words and slang (e.g. `with` -> `wif`).
#
# Spam messages, on the other hand, may be automated to achieve the highest reach possible.
# Thus, message length is allowed to be longer than in a legitimate case, as long as message length restrictions are not reached.
# Additionally, spam messages may use grammatically correct words to infuse a sense of legitimacy.
# Thus, the distribution of spam message lengths may vary less than in the legitimate case.
# Note, however, that some spam messages may be purposefully written with grammatical errors to create a sense of legitimacy (i.e. looks like it is written by a human), such as in the Nigerian prince scam.
#
# So let us extract the message length:

# %%
sms['MessageLength'] = sms.Original_Message.apply(len)
sms[['MessageLength']].value_counts()

# %% [markdown]
# We can see that most messages tend to be short, with some outliers.
# Peculiarly one message is `910` characters long:

# %%
sms[sms.MessageLength == 910]['Original_Message'].iloc[0]

# %% [markdown]
# To get a better sense of the distribution, we plot the message lengths onto a histogram:

# %%
sms[sms.SpamLabel == 'ham'].MessageLength.plot.hist(
    bins=35,
    figsize=(12, 6),
    
    legend=True,
    color=HAM_COLORS['primary'],
    label='Ham messages',
)
sms[sms.SpamLabel == 'spam'].MessageLength.plot.hist(
    legend=True,
    color=SPAM_COLORS['primary'],
    label='Spam messages',
)

plt.xlabel('Message length')

save_figure('message_length_histogram.png')

# %% [markdown]
# Below is a table of related data present as well:

# %%
sms.groupby('SpamLabel')[['MessageLength']].describe()

# %% [markdown]
# As we notice, ham and spam message length peak at different lengths.
#
# Spam messages seem to have a max length of `224`, which could indicate an SMS message length limit.
# Additionally, spam messages seem to be, on average, twice as long as ham messages.
#
# Ham message length, on the other hand, are distributed more heavily.
# This is to be expected from naturally occurring messages, as they may be more tailor-made case-by-case.

# %% [markdown]
# ### Pattern search
#
# As we saw from the worclouds, spam messages seem to include more "call-to-action" words, such as "reply" and "text".
# The intent of spam is often to get the user to perform a certain action, such as visiting a webpage or calling a number.
# This hypothesis is promoted by the word frequencies.
# Thus, we predict that spam messages will, more often, inlcude phone numbers, webpages, and email addresses that a user is supposed to interact with.
# We search messages for related patterns, and flag messages that are considered suspicious (i.e. contain any of the aforementioned texts).

# %%
spam_pattern_candidates = [
    '£',
    '%|€|\$|T&C|@',
    'www|https?',
    'email|sms|free',
    '\d{5,11}',
]


# %%
def message_contains(regex_pattern):
    lower_cased = sms['Original_Message'].str.lower()
    regex_search = lower_cased.str.contains(regex_pattern)
    to_int_mapping = regex_search.map({False: 0, True: 1})
    return to_int_mapping

pattern_search = (
    message_contains(pattern)
    for pattern in spam_pattern_candidates
)

sms['Contains_Special'] = reduce(lambda l, r: l | r, pattern_search)

# %%
sms.head()

# %% [markdown]
# Now that we have the labelling based on pattern matching, let's represent it in a more human-friendly way.
# For this purpose we create a confusion matrix display both false and true negative and positives.

# %%
spam_values = sms['SpamValue']
contains_special_values = sms['Contains_Special']

pattern_search_cf = confusion_matrix(spam_values, contains_special_values)
paint_confusion_matrix(pattern_search_cf)

save_figure('pattern_confusion_matrix.png')

# %% [markdown]
# As can be seen, this feature seems to be a quite good predictor on its own.
# However, we have to keep in mind that it is not perfect, although it may prove fruitful in the analysis section.
#
# The confusion matrix does not, however, show us the ratios between messages containing and not containing the patterns.
# Let's us once more display these ratios as pie charts.

# %%
_, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6) if not OPTIMIZE_ILLUSTRATIONS else (10, 5),
)

ham_special_count = sms[sms.SpamLabel == 'ham']['Contains_Special'].value_counts()
ham_special_count.plot.pie(
    ax=axes[0],
    subplots=True,
    legend=False,
    labels=None,
    ylabel='',
    **PIE_CHART_STYLE
)
axes[0].set_title('Ham pattern ratio')

spam_special_count = sms[sms.SpamLabel == 'spam']['Contains_Special'].value_counts()
spam_special_count.plot.pie(
    ax=axes[1],
    subplots=True,
    legend=False,
    labels=None,
    ylabel='',
    **PIE_CHART_STYLE
)
axes[1].set_title('Spam pattern ratio')

plt.legend(
    ['Excluding pattern', 'Inlcuding pattern'],
    loc='right',
    bbox_to_anchor=(1.29, 0.9) if not OPTIMIZE_ILLUSTRATIONS else (1.3, 0.93)
)

save_figure('pattern_ratios_pie_charts.png')

# %% [markdown]
# ---
#
# ## Analysis
#
# The analysis section of this notebook includes generic machine learning techniques to explore patterns within the test data. The general outline of the section is:
# - Setup
# - Pre-processing
# - Application

# %% [markdown]
# ### Setup
#
# Including all library imports, we have to download the `punkt` dataset from the NLTK library for our tokenization purposes.

# %%
import scipy.stats as stats
import math
import string

from functools import partial, reduce

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import nltk

# %%
nltk.download('punkt')


# %% [markdown]
# ### Pre-processing
#
# The pre-processing pipeline of our machine learning program consists of four steps:
# 1. Punctuation filtering
#     - We filter all punctuation from each message to ensure that tokenization is properly performed.
#       This has the disadvantage that it may remove some sentiments in a message, e.g. compare `haha...` with `haha`.
#       However, deciding which punctuation to be removed requires analysis, which will be performed later.
#       Thus, to scope the program we decided to filter punctuation before.
# 2. Tokenization
#     - Tokenization is the act of splitting a text into its constituent parts, i.e. tokens.
#       In the form of a text message the tokens are separate words.
# 3. Normalizing word case
#     - We normalize the case of characters in each word by lowercasing them, unless the word is already all upper case.
#       This is to ensure that words with the same sentiment are not considered separate, e.g. `Hello` vs. `hello`, which may happen from phone auto capitalization.
#       All upper case words, on the other hand, might carry sentimental value and should thus be left be, e.g. compare `OH NO` with `oh no`.
#       Source: http://sentiment.christopherpotts.net/tokenizing.html#capitalization
# 4. Stop word pruning
#     - We pruned all stop words from the tokenized list of words as they carry no meaning.
#       Stop words are words such as `a`, `and`, and `but`, which do not carry much meaning in generic texts.
#       Removing such might stop learning models from learning wrong patterns and speed up the models in general.

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
        token.lower() if token.upper() != token else token
        for token in tokens
    )

def prune_stopwords(message_contents):
    return (
        word
        for word in message_contents
        if word.lower() not in considered_stopwords
    )

def cast_to_list(generator):
    return list(generator)

message_parsing_pipe = compose(
    filter_punctation,
    tokenize,
    lower_case_unless_all_caps,
    prune_stopwords,
    cast_to_list,
)

sms['Tokens'] = sms['Original_Message'].apply(message_parsing_pipe)

# %% [markdown]
# Next, we count the message occurrence of each token above. This will allow us to later calculate the probability of each word occurring in spam.
#
# For optimization purposes the cell is rather long for such a calculation. Source: https://franekjemiolo.pl/flattening-lists-pandas/

# %%
flattened_tokens = pd.DataFrame(
    [
        (index, value, 1)
        for (index, values) in sms['Tokens'].iteritems()
        for value in values
    ],
    columns=['index', 'Token', 'Count']
).set_index('index')

flattened = sms[['SpamLabel']].join(flattened_tokens)
count = flattened.pivot_table(
    index='Token',
    columns='SpamLabel',
    values='Count',
    aggfunc=np.sum,
    fill_value=0
)
token_data = count.reset_index().rename(columns={'ham': 'nHam', 'spam': 'nSpam'})

# %% [markdown]
# Then calculate the probability of each token occurring in ham. This calculation is simply the ratio between the token occurrence in ham messages over the combined occurrence of the token in both ham and spam:
#
# $$P_{token} = \frac{count_{ham}}{count_{ham} + count_{spam}}$$

# %%
token_data['Probability'] = token_data['nHam'] / (token_data['nSpam'] + token_data['nHam'])
token_data.sort_values(by=['Probability'], ascending=False).head(10)

# %% [markdown]
# We now create the a binary matrix (`identifiers`) where each column is a token and each row is a message.
# The values are `0` in case a token does not exist in a message, and `1` in case the token does.
#
# _NOTE_: the following cells are the heaviest in the notebook, and will take some time.

# %%
identifiers = sms.copy()

# %% [markdown]
# Set each token to `0` for each message.

# %%
for item in token_data['Token'].unique():
    identifiers[item] = 0

# %% [markdown]
# Now set `1` to each matrix cell depending on if the token exists in the message or not.

# %%
for item in token_data['Token'].unique():
    item_search_indices = identifiers['Original_Message'].str.contains(item, regex=False)
    identifiers.loc[item_search_indices, item] = 1

# %% [markdown]
# ### Application
#
# Now that we've pre-processed our dataset properly, we can apply it our machine learning algorithms.
# We are using Naïve Bayes and Random Forest as our two main algorithms.
# Please see the report for more information on the algorithms.

# %% [markdown]
# Before feeding the data into the models, we need to stratify them.
# This means that we split the data into two sets where both have an equal amount of spam and ham messages.
# The split is on a 80 to 20 ratio for training and testing, respectively.

# %%
is_spam = pd.to_numeric(identifiers['SpamValue'])
identfiers_matrix = identifiers.drop(
    ['Original_Message', 'Tokens', 'SpamLabel', 'SpamValue', 'MessageLength', 'Contains_Special'],
    axis=1,
)

X_train, X_test, y_train, y_test = train_test_split(
    identfiers_matrix,
    is_spam,
    test_size=0.2,
    random_state=0,
    stratify=is_spam,
)

# %% [markdown]
# #### Naïve Bayes
#
# Our first model is Naïve Bayes.
# We first train the model with our stratified training data.

# %%
gnb = GaussianNB()
gnb_model = gnb.fit(X_train, y_train)
gnb_predictions = gnb_model.predict(X_test)

# %% [markdown]
# Then we evaluate the model's performance through a confusion matrix.

# %%
gnb_cf = confusion_matrix(y_test, gnb_predictions)
paint_confusion_matrix(gnb_cf)

save_figure('gaussian_confusion_matrix.png')

# %% [markdown]
# #### Random Forest
#
# The other model is a Random Forest model.
# Similarly to the Naïve Bayes model, we start by training the model with the stratified training data.

# %%
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)
clf_predictions = clf.predict(X_test)

# %% [markdown]
# Then we evaluate the model with the test data.

# %%
clf_cf = confusion_matrix(y_test, clf_predictions)
paint_confusion_matrix(clf_cf)

save_figure('random_forest_confusion_matrix.png')

# %% [markdown]
# Just for illustration purposes, the binary matrix is displayed below:

# %%
identfiers_matrix

# %% [markdown]
# ---
#
# ## Demo
#
# For the purposes of the presentation, we included a small demo script.
# The script takes one message and classifies according to the model training outlined above.
#
# For your convenience we've added a couple of sample messages below.
# Run the corresponding cell for the demo you want to run, or create your own!
#
# _Note:_ all cells above have to be run for this to work.

# %% [markdown]
# #### Demo sample message: HAM

# %%
demo_message = "What's for lunch at täffä?"

# %% [markdown]
# #### Demo sample message: SPAM

# %%
demo_message = 'FREE samples claim your prize, call 912-546543'


# %% [markdown]
# #### Demo script
#
# The demo below includes a helper function for printing intermediate processing steps.
# This way you can see the result from each pre-processing step taken.

# %%
def show(prefix_str):
    def verbose_identity(iterable):
        to_list = iterable if type(iterable) is str else list(iterable)
        print(f'{prefix_str}: {to_list}')
        return to_list
    return verbose_identity

print('Pre-processing steps:')
print('-' * 20)

verbose_message_parsing_pipe = compose(
    filter_punctation,
    show('Punctuation pruned'),
    tokenize,
    show('Tokenized'),
    lower_case_unless_all_caps,
    show('Lower-cased'),
    prune_stopwords,
    show('Stopwords pruned'),
)
words = verbose_message_parsing_pipe(demo_message)

print('-' * 20, '\n\n')
print('We predict the input message below to be...\n')
print(f'> {demo_message}')

all_tokens = token_data['Token'].unique()
all_zero_row = [[0] * len(all_tokens)]
demo_binary_matrix = pd.DataFrame(
    all_zero_row,
    columns=all_tokens
)

considered_words = (
    word
    for word in words
    if word in token_data['Token'].unique()
)
for word in considered_words:
    demo_binary_matrix[word] = 1
        
message_prediction = gnb_model.predict(demo_binary_matrix)
if message_prediction[0] == 0:
    print(r'''
     ___  ___  ________  _____ ______      
    |\  \|\  \|\   __  \|\   _ \  _   \    
    \ \  \\\  \ \  \|\  \ \  \\\__\ \  \   
     \ \   __  \ \   __  \ \  \\|__| \  \  
      \ \  \ \  \ \  \ \  \ \  \    \ \  \ 
       \ \__\ \__\ \__\ \__\ \__\    \ \__\
        \|__|\|__|\|__|\|__|\|__|     \|__|
    ''')
else:
    print(r'''
     ________  ________  ________  _____ ______      
    |\   ____\|\   __  \|\   __  \|\   _ \  _   \    
    \ \  \___|\ \  \|\  \ \  \|\  \ \  \\\__\ \  \   
     \ \_____  \ \   ____\ \   __  \ \  \\|__| \  \  
      \|____|\  \ \  \___|\ \  \ \  \ \  \    \ \  \ 
        ____\_\  \ \__\    \ \__\ \__\ \__\    \ \__\ 
       |\_________\|__|     \|__|\|__|\|__|     \|__|
       \|_________|                                  
   ''')
