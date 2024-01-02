#!/usr/bin/env python
# coding: utf-8

# ### 베르누이 나이브베이즈
# - 데이터 : https://www.kaggle.com/team-ai/spam-text-message-classification

# In[1]:


import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv("spam.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data["label"]=data["Category"].map({"spam":1,"ham":0})
data.head()


# In[7]:


X=data["Message"]
y=data["label"]

x_train, x_test, y_train, y_test= train_test_split(X,y,test_size=0.3,
                                                  random_state=103)
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)


# In[10]:


cv=CountVectorizer(max_features=1000, binary=True)
x_train_cv=cv.fit_transform(x_train)


# In[11]:


encoded=x_train_cv.toarray()
encoded


# In[13]:


cv.inverse_transform(encoded[0])


# In[14]:


cv.get_feature_names()


# In[15]:


len(cv.get_feature_names())


# #### 베르누이 나이브베이즈 분류

# In[16]:


nb_clf=BernoulliNB()
nb_clf.fit(x_train_cv, y_train)


# In[18]:


x_test_cv=cv.fit_transform(x_test)


# In[19]:


encoded2=x_test_cv.toarray()
encoded2


# In[20]:


pred=nb_clf.predict(x_test_cv)


# In[21]:


accuracy_score(y_test, pred)


# ### < 과제 >
# - 베르누이 나이브베이즈 분류 모델을 사용하여 스팸 메세지 분류
# - 정제, 필터링 작업 후 분류하여 성능확인

# In[24]:


text=cv.get_feature_names()


# In[25]:


from nltk.tokenize import word_tokenize
from nltk. corpus import stopwords
from collections import Counter

vocab=Counter()

sentences=[]
stop_words=set(stopwords.words('english'))

for i in text:
    sentence=word_tokenize(i)
    result=[]
    
    for word in sentence:
        word=word.lower()
        if word not in stop_words:
            if len(word)>2:
                result.append(word)
                vocab[word]=vocab[word]+1
    sentences.append(result)
    
print(sentences)


# In[ ]:




