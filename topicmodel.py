#!/usr/bin/env python
# coding: utf-8

# ### 구매 후기를 이용한 토픽모델링

# - 데이터 : https://drive.google.com/file/d/1eeTHELYDR0UW9CK7yODGhcUGylTfpLv4/view?usp=sharing

# In[3]:


import pandas as pd

review=pd.read_csv("centrum_review.txt", header=None)
review


# In[4]:


from konlpy.tag import Okt

okt=Okt()


# In[5]:


docs=[]
for i in review[0]:
    docs.append(okt.nouns(i))


# In[6]:


docs


# In[7]:


review.loc[5]


# In[8]:


def get_nouns(text):
    nouns=okt.nouns(text)
    nouns=[word for word in nouns if len(word)>1]
    return nouns


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(tokenizer=get_nouns)

tdm=cv.fit_transform(review[0])


# In[10]:


cv.get_feature_names()[:10]


# In[11]:


cv.vocabulary_


# In[12]:


doc=tdm[0].toarray()
doc


# In[13]:


count=tdm.sum(axis=0)
count


# In[14]:


word_count=pd.DataFrame({"단어":cv.get_feature_names(),
                        "빈도":count.flat})

word_count.head()


# In[15]:


word_count.sort_values(by="빈도",ascending=False)


# In[16]:


import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_path="C:/Windows/Fonts/malgun.ttf"
font=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

word_count_index=word_count.copy()
word_count_index.index=word_count_index["단어"]
word_count_index.sort_values(by="빈도", ascending=False)[:10].plot.bar()
plt.show()


# In[17]:


docs_noun=docs.copy()

for i in range(len(docs_noun)):
    docs_noun[i]=' '.join(docs_noun[i])
    
docs_noun


# In[18]:


noun_doc=' '.join(docs_noun)
noun_doc=noun_doc.strip()
noun_doc


# In[19]:


from wordcloud import WordCloud

font_path="C:/Windows/Fonts/malgun.ttf"

wc=WordCloud(font_path=font_path, background_color="white")
wc.generate(noun_doc)

plt.figure()
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[20]:


import gensim
from gensim import corpora, models

resultList=[]
keyword=5

texts=[]
for line in docs:
    tokens=[word for word in line if len(word)>1]
    texts.append(tokens)
    
dictionary=corpora.Dictionary(texts)
corpus=[dictionary.doc2bow(text) for text in texts]

ldamodel= models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=5)

for num in range(5):
    resultList.append(ldamodel.show_topic(num, keyword))


# In[21]:


resultList


# In[22]:


ldamodel.print_topics(num_words=5)


# In[23]:


ldamodel.get_document_topics(corpus)[0]


# In[ ]:




