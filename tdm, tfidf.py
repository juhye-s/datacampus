#!/usr/bin/env python
# coding: utf-8

# ## TDM

# corpus = ['you know I want your love',
#           'I like you',
#           'what should I do']

# In[3]:


from sklearn.feature_extraction.text import CountVectorizer

corpus=['you know i want your love','i like you','what should i do']

vector=CountVectorizer()

print(vector.fit_transform(corpus).toarray())
print(vector.vocabulary_)


# In[3]:


import pandas as pd
df=pd.read_csv('폭염.csv',sep=",",quotechar='"',
               error_bad_lines=False, encoding='utf-8')


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df['본문'].head()


# In[8]:


from konlpy.tag import Okt
okt=Okt()


# In[9]:


okt.nouns('아침에 해를 보며 버스를 탔다')


# In[10]:


import re


# In[11]:


def get_nouns(text):
    nouns=okt.nouns(text)
    nouns=[word for word in nouns if len(word)>1]
    nouns=[word for word in nouns if not re.match(r'\d+',word)]
    return nouns


# In[12]:


get_nouns('아침에 해를 보며 버스를 탔다')


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer


# In[14]:


cv=CountVectorizer(max_features=1000, tokenizer=get_nouns)


# In[15]:


tdm=cv.fit_transform(df['본문'])


# In[16]:


words=cv.get_feature_names()


# In[17]:


words[:10]


# In[18]:


cv.vocabulary_


# In[19]:


cv.vocabulary_['가격']


# In[21]:


doc=tdm[0].toarray()
doc


# In[23]:


count=tdm.sum(axis=0)
print(count)


# In[24]:


import pandas as pd


# In[25]:


word_count=pd.DataFrame({'단어':cv.get_feature_names(),'빈도':count.flat})


# In[27]:


word_count.tail()


# In[28]:


sorted_df=word_count.sort_values('빈도',ascending=False)


# In[29]:


sorted_df.head(10)


# In[32]:


import matplotlib.pyplot as plt

import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus']=False

path="C:/windows/Fonts/malgun.ttf"
font_name=font_manager.FontProperties(fname=path).get_name()
rc('font', family=font_name)

plt.barh(sorted_df.head(20)["단어"],sorted_df.head(20)["빈도"])
plt.show()


# ## TF-IDF

# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer

corpus=['you know i want ypur love',
       'i like you', 'what should i do']
tfidfv=TfidfVectorizer().fit(corpus)

print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)


# In[ ]:





# In[ ]:





# ## < 과제 >
# - 폭염 관련 뉴스 메타데이터 TF-IDF 

# In[35]:


import pandas as pd
df=pd.read_csv('폭염.csv',sep=",",quotechar='"',
               error_bad_lines=False, encoding='utf-8')
tfidfv=TfidfVectorizer().fit(df)
print(tfidfv.transform(df).toarray())
print(tfidfv.vocabulary_)


# In[ ]:




