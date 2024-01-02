#!/usr/bin/env python
# coding: utf-8

# ### RNN, 스팸분류
# - 데이터 : https://www.kaggle.com/team-ai/spam-text-message-classification

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("spam.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data["label"]=data["Category"].map({"spam":1,"ham":0})
data.head()


# In[7]:


data['Message'].nunique(),data['label'].nunique()


# In[8]:


data.drop_duplicates(subset=['Message'], inplace=True)


# In[9]:


data.info()


# In[10]:


import matplotlib.pyplot as plt

data['label'].value_counts().plot(kind="bar")
plt.show()


# In[11]:


X_data=data["Message"]
y_data=data["label"]


# In[14]:


from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_data)

sequences=tokenizer.texts_to_sequences(X_data)


# In[17]:


print(sequences[0])


# In[15]:


word_index=tokenizer.word_index
print(word_index)


# In[16]:


print(len(word_index))


# In[18]:


X_data=sequences

print("최대길이:",max(len(l) for l in X_data))
print("평균길이:",(sum(map(len,X_data))/len(X_data)))


# In[19]:


from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[20]:


vocab_size= len(word_index)+1
max_len=189

data=pad_sequences(X_data, maxlen=max_len)
print("data shape:", data.shape)


# In[21]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(data, y_data,
                                                 test_size=0.3,
                                                  random_state=103)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[22]:


model=Sequential()
model.add(Embedding(vocab_size,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))


# In[23]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history=model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)


# In[24]:


model.evaluate(x_test, y_test)


# In[27]:


epochs=range(1,len(history.history['acc'])+1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()


# ##과제 자유주제로 기사를 크롤링 한 후, RNN의 언어 모델링을 이용해 키워드로 첫단어로 하여 20단어 생성하기

# In[ ]:




