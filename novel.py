#!/usr/bin/env python
# coding: utf-8

# ### 소설 쓰기
# - BEXX0003.txt

# In[1]:


import codecs
from bs4 import BeautifulSoup


# In[3]:


fp=codecs.open("BEXX0003.txt","r",encoding="utf-16")
soup=BeautifulSoup(fp,'html.parser')
body=soup.select_one("body > text")
text=body.getText()


# In[5]:


from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras.utils.data_utils import get_file

import random
import sys
import re


# In[7]:


text=re.sub(r'<.*>', '', text)
text=re.sub(r'\n', ' ', text)
text=re.sub(r' +', ' ', text)

print('corpus length:', len(text))


# In[8]:


type(text)


# In[9]:


chars=sorted(list(set(text)))
chars[:100]


# In[10]:


print('total chars:', len(chars))

char_indices=dict((c,i) for i,c in enumerate(chars))
indices_char=dict((i,c) for i,c in enumerate(chars))


# In[11]:


maxlen=40
step=3
sentences=[]
next_chars=[]

for i in range(0,len(text)-maxlen, step):
    sentences.append(text[i: i+ maxlen])
    next_chars.append(text[i+maxlen])
    
print('sequences:', len(sentences))


# In[13]:


import numpy as np


# In[14]:


x=np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y=np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i,t,char_indices[char]]=1
    y[i,char_indices[next_chars[i]]]=1    


# In[16]:


model=Sequential()

model.add(LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])


# In[17]:


def sample(preds, temperature=1.0):
    preds=np.asarray(preds).astype('floar64')
    preds=np.log(preds)/ temperature
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1, preds,1)
    return np.argmax(probas)


# In[20]:


def on_epoch_end(epoch,_):
    print('\n-----Generationg text after Epoch: %d'% epoch)
    
    start_index=np.random.randint(0,len(text)-maxlen-1)
    
    
    generated=' '
    sentence=text[start_index:start_index+maxlen]
    generated+=sentence
    print('-------Generation with seed:" '+sentence+ '"')
    sys.stdout.write(generated)
    
    for i in range(400):
        x_pred=np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0,t,char_indices[char]]=1.
            
            preds=model.predict(x_pred, verbose=0)[0]
            next_index=sample(preds,0.5)
            next_char=indices_char[next_index]
            
            generated+=next_char
            sentence=sentence[1:]+next_char
            
            sys.stdout.write(next_char)
            sys.stdout.flush()
            
        print()
        
print_callback=LambdaCallback(on_epoch_end=on_epoch_end)


# In[21]:


model.fit(x, y, batch_size=128, epochs=60, callbacks=[print_callback])


# In[ ]:





# In[ ]:





# In[ ]:




