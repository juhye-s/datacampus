#!/usr/bin/env python
# coding: utf-8

# ### Word2vec

# -"http://www.gutenberg.org/files/2591/2591-0.txt "

# In[29]:


import requests
import re


# In[30]:


req=requests.get("http://www.gutenberg.org/files/2591/2591-0.txt")


# In[31]:


grimm=req.text[2993:540077]


# In[32]:


grimm


# In[33]:


grimm=re.sub(r'[^\w\.]'," ",grimm)


# In[34]:


grimm


# In[35]:


from nltk.tokenize import sent_tokenize


sentence=sent_tokenize(grimm)
print(sentence[0])


# In[40]:


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


# In[41]:


pip install gensim


# In[42]:


from gensim.models.word2vec import Word2Vec


# In[43]:


model=Word2Vec(data_cln, sg=1, vector_size=100,
              window=3, min_count=3, workers=-1)


# In[44]:


model.save("word2vec.model")


# In[45]:


model=Word2Vec.load("word2vec.model")


# In[46]:


model.wv["apple"]


# In[48]:


model.wv.similarity("apple","bird")


# In[49]:


model.wv.similarity("princess","king")


# In[50]:


model.wv.similarity("prince","queen")


# In[51]:


model.wv.most_similar("apple")


# In[53]:


model.wv.most_similar(positive=["queen","princess"], negative=["king"])


# In[ ]:





# In[ ]:




