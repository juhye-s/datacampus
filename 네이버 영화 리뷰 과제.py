#!/usr/bin/env python
# coding: utf-8

# ##과제 네이버 영화리뷰를 이용하여 Word2Vec 생성

# In[9]:


import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from gensim.models.word2vec import Word2Vec
import pandas as pd
import requests
import re


# In[10]:


df=pd.read_csv('ratings.txt',sep="\t",header=None)
df.head()


# In[37]:


okt=Okt()


# In[38]:


doc=[]

for i in df[1]:
    doc.append(okt.nouns(i))
    doc.append(okt.abject(i))
    
for i in range(len(doc)):
    doc[i]=' '.join(doc[i])
    
doc[0]


# In[ ]:


from nltk.tokenize import sent_tokenize


sentence=sent_tokenize(grimm)
print(sentence[0])


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

data_cln=[]
stop_words=set(stopwords.words('english'))

for i in sentence:
    sentence=word_tokenize(i)
    result=[]
    
    for word in sentence:
        word=word.lower()
        if word not in stop_words:
            if len(word)>2:
                result.append(word)
    data_cln.append(result)
    
print(data_cln[0])


# In[ ]:


model=Word2Vec(data_cln, sg=1, vector_size=100,
              window=3, min_count=3, workers=-1)


# In[ ]:


model.save("word2vec.model")


# In[ ]:


model=Word2Vec.load("word2vec.model")


# In[ ]:





# In[ ]:





# In[ ]:




