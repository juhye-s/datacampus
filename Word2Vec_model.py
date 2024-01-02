#!/usr/bin/env python
# coding: utf-8

# ### 미리 학습된 모델 사용

# #### 영어
# - 데이터 : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

# In[2]:


from gensim.models import KeyedVectors


# In[7]:


#다운로드 오류...

model=KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                       binary=True)


# In[8]:


model.vectors.shape


# In[ ]:


print(model.similarity('this','is'))
print(model.similarity('post','book'))


# In[ ]:


model.most_similar("book")


# In[ ]:


print(model.similarity('apple','forest'))


# In[ ]:


model.most_similar("foreset")


# In[ ]:




