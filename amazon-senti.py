#!/usr/bin/env python
# coding: utf-8

# ### 감성사전구축

# - 데이터 : https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('amazon_cells_labelled.txt',sep="\t",header=None)
df.head()


# In[3]:


df.info()


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


X=df[0]
y=df[1]


# In[6]:


tfidf=TfidfVectorizer(stop_words='english', max_features=1000)
X_tdm=tfidf.fit_transform(X)


# In[7]:


X_tdm.toarray()


# In[8]:


tfidf.get_feature_names()


# In[9]:


tfidf.inverse_transform(X_tdm[0])


# In[11]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(X_tdm, y, test_size=0.3,
                                                  random_state=103)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[32]:


from sklearn.linear_model import LogisticRegressionCV


# In[33]:


lr_clf=LogisticRegression()
lr_clf.fit(x_train, y_train)


# In[34]:


lr_clf.score(x_train, y_train)


# In[35]:


pred=lr_clf.predict(x_test)


# In[36]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)


# In[37]:


lr_clf.coef_


# In[38]:


st_df=pd.DataFrame({"단어":tfidf.get_feature_names(),
                   "회귀계수":lr_clf.coef_.flat})
st_df.tail()


# In[39]:


st_neg=st_df[st_df["회귀계수"]<0].sort_values("회귀계수")
st_neg.head()


# In[40]:


st_pos=st_df[st_df["회귀계수"]>0].sort_values("회귀계수",ascending= False)
st_pos.head()


# In[41]:


import numpy as np

st_df["극성"]=np.sign(st_df["회귀계수"])

st_df.tail()


# In[42]:


st_df["극성"].value_counts()


# In[43]:


st_df["극성"].sum()/st_df["극성"].abs().sum()


# ### < 과제 >
# - TFIDF 이용해서 네이버 영화 평점 감성 사전 구축
#     - ratings.txt 
# - 극성 구하기
# 

# In[45]:


import pandas as pd


# In[60]:


rt=pd.read_csv('ratings.txt',sep="\t", header=0,
              error_bad_lines=False, encoding='utf-8')
rt.head()


# In[61]:


rt.info()


# In[63]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[64]:


X=df[0]
y=df[1]


# In[75]:


tfidf=TfidfVectorizer()
X_tdm=tfidf.fit_transform(X)


# In[76]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(X_tdm, y, test_size=0.3,
                                                  random_state=103)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[77]:


from sklearn.linear_model import LogisticRegressionCV


# In[78]:


lr_clf=LogisticRegression()
lr_clf.fit(x_train, y_train)


# In[79]:


lr_clf.score(x_train, y_train)


# In[80]:


pred=lr_clf.predict(x_test)


# In[81]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)


# In[82]:


rt_df=pd.DataFrame({"단어":tfidf.get_feature_names(),
                   "회귀계수":lr_clf.coef_.flat})
rt_df.tail()


# In[83]:


import numpy as np

rt_df["극성"]=np.sign(rt_df["회귀계수"])

rt_df.tail()


# In[84]:


rt_df["극성"].sum()/rt_df["극성"].abs().sum()


# In[ ]:




