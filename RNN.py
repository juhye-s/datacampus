#!/usr/bin/env python
# coding: utf-8

# ### 언어모델링

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN


# In[2]:


train_X=[[0.1,4.2,1.5,1.1,2.8],[1.0,3.1,2.5,0.7,1.1],
         [0.3,2.1,1.5,2.1,0.1],[2.2,1.4,0.5,0.9,1.1]]

print(np.shape(train_X))


# In[6]:


train_X=[[[0.1,4.2,1.5,1.1,2.8],[1.0,3.1,2.5,0.7,1.1],
         [0.3,2.1,1.5,2.1,0.1],[2.2,1.4,0.5,0.9,1.1]]]
train_X=np.array(train_X, dtype=np.float32)

print(train_X)
print(train_X.shape)


# In[7]:


rnn=SimpleRNN(3, return_sequences=True, return_state=True)

rnn(train_X)
print(rnn(train_X))


# In[9]:


hidden_states, last_states=rnn(train_X)

print('train_X:{}, shape:{}'.format(train_X, train_X.shape))
print('hidden states:{}, shape:{}'.format(hidden_states, hidden_states.shape))
print('last hidden state:{}, shape:{}'.format(last_states, last_states.shape))


# In[11]:


train_X=[[[0.1,4.2,1.5,1.1,2.8],[1.0,3.1,2.5,0.7,1.1],
         [0.3,2.1,1.5,2.1,0.1],[2.2,1.4,0.5,0.9,1.1]]]
train_X=np.array(train_X, dtype=np.float32)

print(train_X)
print(train_X.shape)


# In[12]:


from tensorflow.keras.layers import SimpleRNN, Dense

rnn=SimpleRNN(3,return_sequences=True, return_state=True)
rnn=Dense(5, activation="softmax")

print(rnn(train_X))


# In[13]:


text="이 여름 다시 한번 설레고 싶다 그 여름 틀어줘"


# In[14]:


from keras_preprocessing.text import Tokenizer

t=Tokenizer()
t.fit_on_texts([text])

encoded=t.texts_to_sequences([text])[0]


# In[15]:


t.texts_to_sequences([text])


# In[16]:


t.texts_to_sequences([text])[0]


# In[18]:


vocab_size=len(t.word_index)+1

print('단어 집합의 크기:',vocab_size)


# In[19]:


print(t.word_index)


# In[20]:


sequences=list()

for c in range(1, len(encoded)):
    sequence=encoded[c-1:c+1]
    sequences.append(sequence)
    
print('단어 묶음의 개수:',len(sequences))


# In[21]:


import numpy as np

X, y= zip(*sequences)

X=np.array(X)
y=np.array(y)


# In[22]:


print(X)
print(y)


# In[23]:


from tensorflow.keras.utils import to_categorical

y=to_categorical(y, num_classes=vocab_size)

print(y)


# In[25]:


from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.models import Sequential

model=Sequential()
model.add(Embedding(vocab_size, 9, input_length=1))
model.add(SimpleRNN(9))
model.add(Dense(vocab_size, activation='softmax'))


# In[26]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y, epochs=500)


# In[27]:


print(t.word_index.items())


# In[28]:


def predict_next_word(model, t, current_word):
    encoded=t.texts_to_sequences([current_word])[0]
    encoded=np.array(encoded)
    result=model.predict_classes(encoded, verbose=0)
    
    for word, index in t.word_index.items():
        if index==result:
            return word


# In[32]:


print(predict_next_word(model, t,'여름'))


# In[42]:


def sentence_generation(model, t, current_word, n):
    init_word=current_word
    sentence=''
    
    for _ in range(n):
        encoded=t.texts_to_sequences([current_word])[0]
        encoded=np.array(encoded)
        result=model.predict_classes(encoded, verbose=0)
        
        for word, index in t.word_index.items():
            if index==result:
                break
        current_word=word
        sentence=sentence+' '+word
            
        
    wentence=init_word+ sentence
    return sentence
    


# In[43]:


print(sentence_generation(model,t,'여름',6))


# In[44]:


print(sentence_generation(model,t,'다시',6))


# In[45]:


text="""경마장에 있는 말이 뛰고 있다. 그의 말이 법이다. 가는 말이 고와야 오는 말이 곱다."""


# In[46]:


import tensorflow as tf
from keras_preprocessing.text import Tokenizer

t=Tokenizer()
t.fit_on_texts([text])

encoded=t.texts_to_sequences([text])[0]


# In[47]:


vocab_size=len(t.word_index)+1

print('단어 집합의 크기: %d'%vocab_size)


# In[48]:


print(t.word_index)


# In[52]:


sequences=list()

for line in text.split('.'):
    encoded=t.texts_to_sequences([line])[0]
    for i in range(1,len(encoded)):
        sequence=encoded[:i+1]
        sequences.append(sequence)
        
print('훈련 데이터의 개수:',len(sequences))


# In[53]:


print(sequences)


# In[54]:


print(max(len(l) for l in sequences ))


# In[56]:


from keras.preprocessing.sequence import pad_sequences

sequences=pad_sequences(sequences, maxlen=6, padding='pre')


# In[57]:


print(sequences)


# In[58]:


import numpy as np 
sequences=np.array(sequences)

X=sequences[:,:-1]
y=sequences[:,-1]


# In[59]:


print(X)


# In[60]:


print(y)


# In[61]:


from tensorflow.keras.utils import to_categorical

y=to_categorical(y, num_classes=vocab_size)


# In[62]:


print(y)


# In[63]:


from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.models import Sequential

model=Sequential()
model.add(Embedding(vocab_size, 10, input_length=5))
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X,y, epochs=200)


# In[67]:


def sentence_generation(model, t, current_word, n):
    init_word=current_word
    sentence=''
    
    for _ in range(n):
        encoded=t.texts_to_sequences([current_word])[0]
        encoded=pad_sequences([encoded],maxlen=5, padding='pre')
        result=model.predict_classes(encoded, verbose=0)
        
        for word, index in t.word_index.items():
            if index==result:
                break
        current_word= current_word +' '+ word
        sentence=sentence+' '+word
            
        
    sentence=init_word+ sentence
    return sentence
    


# In[68]:


print(sentence_generation(model,t,'경마장에',4))


# In[69]:


print(sentence_generation(model,t,'그의',2))


# In[70]:


print(sentence_generation(model,t,'경마장에',7))


# In[ ]:




