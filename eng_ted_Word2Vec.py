#!/usr/bin/env python
# coding: utf-8

# - 데이터 : ted_en-20160408.xml 

# In[3]:


import gensim


# In[4]:


import re
from lxml import etree
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

targetXML=open('ted_en-20160408.xml','r',encoding='utf-8')

target_text=etree.parse(targetXML)
parse_text='\n'.join(target_text.xpath('//content/text()'))


# In[6]:


content_text=re.sub(r'\([^)]*\)','',parse_text)


# In[7]:


sent_text=sent_tokenize(content_text)


# In[8]:


normalized_text=[]

for string in sent_text:
    tokens=re.sub(r"[^a-z0-9]+"," ",string.lower())
    normalized_text.append(tokens)


# In[9]:


normalized_text


# In[11]:


result=[]
result=[word_tokenize(sentence) for sentence in normalized_text]


# In[12]:


print(result[0])


# In[15]:


from gensim.models.word2vec import Word2Vec
model= Word2Vec(result, vector_size=100,
              window=10, min_count=10, workers=-1, sg=0)


# In[16]:


model.wv.similarity("man","woman")


# In[17]:


model.wv.similarity("man","boy")


# In[19]:


model.wv.most_similar("man")


# In[20]:


from nltk.corpus import stopwords

data_cln=[]
stop_words=set(stopwords.words('english'))

for i in normalized_text:
    sentence=word_tokenize(i)
    result=[]
    
    for word in sentence:
        if word not in stop_words:
            if len(word)>2:
                result.append(word)
    data_cln.append(result)
    
print(data_cln[0])


# In[21]:


model2= Word2Vec(data_cln, vector_size=100,
              window=10, min_count=10, workers=-1, sg=0)


# In[23]:


model2.wv.most_similar("man")


# In[ ]:




