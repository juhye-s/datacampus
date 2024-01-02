#!/usr/bin/env python
# coding: utf-8

# ### 토지

# - 데이터 : https://ithub.korean.go.kr/user/total/database/corpusManager.do

# In[1]:


import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from gensim.models.word2vec import Word2Vec


# In[3]:


fp=codecs.open("BEXX0003.txt","r",encoding="utf-16")
soup=BeautifulSoup(fp,"html.parser")
body=soup.select_one("body > text")
text=body.getText()


# In[4]:


body


# In[5]:


text


# In[8]:


okt=Okt()
ex1=okt.pos("아버지가방에들어가신다.",norm=True, stem=True)
ex2=okt.pos("아버지가방에들어가신다.")
print(ex1)
print(ex2)


# In[9]:


ex3=okt.pos("그래욬ㅋ?")
ex4=okt.pos("그래욬ㅋ?",norm=True, stem=True)
ex5=okt.pos("그래욬ㅋ?",norm=False, stem=True)
ex6=okt.pos("그래욬ㅋ?",norm=True, stem=False)
print(ex3)
print(ex4)
print(ex5)
print(ex6)


# In[14]:


okt=Okt()
results=[]
lines=text.split("\n")

for line in lines:
    malist=okt.pos(line, norm=True, stem=True)
    r=[]
    
    for word in malist:
        if not word[1] in ["Josa","Eomi","Punctuation"]: #if word[1]=="None"(명사만 가져오겠다)
            r.append(word[0])
            
        results.append(r)


# In[15]:


print(results)


# In[16]:


model=Word2Vec(results, vector_size=200, window=10, min_count=2, sg=1)


# In[ ]:


model.wv,similarity("아버지","집")


# In[ ]:


model.wv,similar(["집"])


# In[ ]:


model.wv,similar(["아버지"])


# In[ ]:




