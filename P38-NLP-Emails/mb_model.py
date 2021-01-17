# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:07:19 2020

@author: deshp
"""
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from wordcloud import WordCloud
import re # for handling string
import string # for handling mathematical operations
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from imblearn.combine import SMOTEENN 
from joblib import dump
import joblib

df= pd.read_csv("emails.csv")
# creating new dataframe using "content" and "class"
df1= df.iloc[:,3:5]

duplicate= df[df1.duplicated()] 
df1= df1.drop_duplicates() 

# text cleaning
df1['cleaned']=df1['content'].apply(lambda x: x.lower()) # remove lower cases
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('\w*\d\w*','', x)) # remove digits and words with digits
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x)) # remove punctuation
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('\n'," ",x)) # remove extra spaces
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ',x)) # remove special characters
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub(r"http\\S+", " ",x)) # remove hyperlinks
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub(' +',' ',x)) # remove extra spaces
df1['cleaned']=df1['cleaned'].apply(lambda x: x.split('\n\n')[0])
df1['cleaned']=df1['cleaned'].apply(lambda x: x.split('\n')[0])
df1['cleaned'].head()

# tokenise entire df
def identify_tokens(row):
    new = row['cleaned']
    tokens = nltk.word_tokenize(new)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df1['cleaned'] = df1.apply(identify_tokens, axis=1)

#lemmatization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

df1['lemma'] = df1['cleaned'].apply(lemmatize_text)

# remove stopwords
#nltk.download('stopwords')
stop_words = []
with open("stop.txt",encoding="utf8") as f:
    stop_words = f.read()
    
# getting list of stop words
stop_words = stop_words.split("\n")               

def remove_stops(row):
    my_list = row['lemma']
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)

df1['lemma_meaningful'] = df1.apply(remove_stops, axis=1)
df1['lemma_meaningful'].tail()

# rejoin meaningful stem words in single string like a sentence
def rejoin_words(row):
    my_list = row['lemma_meaningful']
    joined_words = ( " ".join(my_list))
    return joined_words

df1['final'] = df1.apply(rejoin_words, axis=1)

spam= ' '.join(list(df1[df1['Class'] == "Abusive"]['final']))
ham= ' '.join(list(df1[df1['Class'] == "Non Abusive"]['final']))

# # Preparing email texts into word count matrix format 
mail= df1.loc[:,['final','Class']]
mail['final'].replace('', np.nan, inplace=True)
mail.dropna(subset=['final'], inplace=True)
mail['label'] = mail['Class'].map({'Abusive': 0, 'Non Abusive': 1})

#create vectors from words
cv = CountVectorizer(max_features=5000).fit(mail.final)

# vectorising all mails
all_emails_matrix = cv.transform(mail['final'])
# dtm
df_dtm = pd.DataFrame(all_emails_matrix.toarray(), columns=cv.get_feature_names())
df_dtm.index=mail.index

# handle imbalance
sme = SMOTEENN(random_state=42)
x_res, y_res = sme.fit_resample(all_emails_matrix, mail['label'])
z=y_res.to_frame()

# splitting data into train and test data sets 
pd.DataFrame(x_res.todense(), columns=cv.get_feature_names())
pd.DataFrame(x_res.todense()[y_res == 'Abusive'], columns= cv.get_feature_names())

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x_res,y_res,test_size=0.3)

## Without TFIDF matrices # # Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(x_res)

# Preparing TFIDF for train n test emails
train_tfidf = tfidf_transformer.transform(x_train)
test_tfidf = tfidf_transformer.transform(x_test)

# Multinomial Naive Bayes
clf = MB()
clf.fit(train_tfidf,y_train)
train_pred_m_tfidf = clf.predict(train_tfidf)
accuracy_train_m_tfidf = np.mean(train_pred_m_tfidf==y_train) # 97.15

test_pred_m_tfidf = clf.predict(test_tfidf)
accuracy_test_m_tfidf = np.mean(test_pred_m_tfidf==y_test) # 97.89

#filename='nlp_model.pkl'
joblib.dump(clf,'nlp_model.pkl')
joblib.dump(cv,'vector.pkl')

# filename='nlp_model.pkl'
# pickle.dump(clf,open(filename,'wb'))
# pickle.dump(cv,open('vector.pkl','wb'))