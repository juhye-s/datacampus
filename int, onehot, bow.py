#!/usr/bin/env python
# coding: utf-8

# ## 정수인코딩

# In[7]:


text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."


# In[11]:


from nltk.tokenize import sent_tokenize

text=sent_tokenize(text)
print(text)


# In[12]:


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


# In[13]:


print(vocab)


# In[15]:


vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)
print(vocab_sorted)


# In[17]:


word_to_index={}
i=0
for (word, frequency) in vocab_sorted:
    if frequency>1:
        i=i+1
        word_to_index[word]=i
print(word_to_index)


# ## one-hot encoding

# In[18]:


example = "이 여름 다시 한번 설레고 싶다 그때 그 여름을 틀어줘 그 여름을 들려줘 이 여름도 언젠가는 그해 여름 오늘이 가장 젊은 내 여름"


# In[19]:


from konlpy.tag import Okt
okt=Okt()

token=okt.morphs(example)

print(token)


# In[26]:


word_to_index={}

for voca in token:
    if voca not in word_to_index.keys():
        word_to_index[voca]=len(word_to_index)
        
print(word_to_index)


# In[29]:


def one_hot_encoding(word, word_to_index):
    one_hot_vector=[0]*(len(word_to_index))
    index=word_to_index[word]
    one_hot_vector[index]=1
    return one_hot_vector


# In[30]:


one_hot_encoding("여름",word_to_index)


# In[31]:


one_hot_encoding("가장",word_to_index)


# ### < 과제 >
# - 1글자 단어 제외하고 one hot 벡터 만들기

# In[56]:


from konlpy.tag import Okt
okt=Okt()

token=okt.morphs(example)

print(token)


# In[58]:


word_to_index={}

for voca in token
    if voca not in word_to_index.keys():
        if len(voca)>2:
            word_to_index[voca]=len(word_to_index)
        
print(word_to_index)


# ### BoW

# In[32]:


text = """이 여름 다시 한번 설레고 싶다. 그때 그 여름을 틀어줘. 그 여름을 들려줘. 이 여름도 언젠가는 그해 여름. 오늘이 가장 젊은 내 여름."""


# In[59]:


from konlpy.tag import Okt
import re

okt=Okt()


# In[60]:


token=re.sub("(\.)","",text)
print(token)


# In[61]:


token=okt.morphs(token)

print(token)


# In[62]:


word2index={}
bow=[]
for voca in token:
    if voca not in word2index.keys():
        word2index[voca]=len(word2index)
        bow.insert(len(word2index)-1,1)
        
    else:
        index=word2index.get(voca)
        bow[index]=bow[index]+1
        
print(word2index)


# In[63]:


bow


# In[65]:


from sklearn.feature_extraction.text import CountVectorizer

corpus=['you know i want your love. because i love you.']
vector=CountVectorizer()

print(vector.fit_transform(corpus).toarray())
print(vector.vocabulary_)


# In[68]:


from sklearn.feature_extraction.text import CountVectorizer

text=['you know i want your love. because i love you.']
vect=CountVectorizer(stop_words=["you","your"])

print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)


# In[70]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

sw=stopwords.words("english")

text=['you know i want your love. because i love you.']
vect=CountVectorizer(stop_words=sw)

print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)


# In[ ]:





# In[ ]:





# In[ ]:




