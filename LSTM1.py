#!/usr/bin/env python
# coding: utf-8

# ### LSTM 문장생성하기
# - https://www.kaggle.com/aashita/nyt-comments
# - ArticlesApril2018.csv

# In[5]:


import pandas as pd

df=pd.read_csv('ArticlesApril2018.csv')
df.head()


# In[6]:


df.info()


# In[3]:


print(df.columns)
print('열 개수:',len(df.columns))


# In[4]:


df['headline'].isnull().values.any()


# In[7]:


headline=[]
headline.extend(list(df.headline.values))
headline[:5]


# In[8]:


len(headline)


# In[12]:


headline=[n for n in headline if n != "Unknown"]
len(headline)


# In[13]:


headline[:5]


# In[16]:


from string import punctuation
def repreprocessing(s):
    s=s.encode("utf8").decode("ascii","ignore")
    return ''.join(c for c in s if c not in punctuation).lower()

text=[repreprocessing(x) for x in headline]
text[:5]


# In[17]:


from keras_preprocessing.text import Tokenizer

t=Tokenizer()
t.fit_on_texts(text)

vocab_size=len(t.word_index)+1
print('단어 집합의 크기: %d'% vocab_size)


# In[18]:


sequences=list()

for line in text:
    encoded=t.texts_to_sequences([line])[0]
    for i in range(1,len(encoded)):
        sequence=encoded[:i+1]
        sequences.append(sequence)
        
sequences[:11]
        


# In[20]:


index_to_word={}
for key, value in t.word_index.items():
    index_to_word[value]=key
    
index_to_word[582]


# In[21]:


max_len=max(len(l) for l in sequences)
print(max_len)


# In[22]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

sequences=pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences[:3])


# In[24]:


import numpy as np

sequences=np.array(sequences)

X=sequences[:,:-1]
y=sequences[:,-1]


# In[25]:


print(X[:3])


# In[26]:


print(y[:3])


# In[27]:


from tensorflow.keras.utils import to_categorical

y=to_categorical(y, num_classes=vocab_size)


# In[30]:


from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential


model=Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y, epochs=200)


# In[32]:


def sentence_generation(model, t, current_word, n):
    init_word=current_word
    sentence=' '
    for _ in range(n):
        encoded=t.texts_to_sequences([current_word])[0]
        encoded=pad_sequences([encoded], maxlen=23, padding='pre')
        result=model.predict_classes(encoded, verbose=0)
        
        for word, index in t.word_index.items():
            if index==result:
                break
        current_word=current_word+' '+word
        sentence=sentence+' '+word
        
    sentence=init_word+sentence
    return sentence
                


# In[33]:


print(sentence_generation(model, t, 'i',10))


# In[34]:


print(sentence_generation(model, t, 'how',10))


# In[ ]:





# In[ ]:




