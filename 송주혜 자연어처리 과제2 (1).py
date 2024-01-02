#!/usr/bin/env python
# coding: utf-8

# In[18]:


import konlpy


# 분석기 비교하기

# In[8]:


string=input("분석할 텍스트를 입력하세요:")


# In[12]:


from konlpy.tag import Kkma
kma=Kkma()
result=kma.pos(string)

for lex, pos in result:
    print("{}\t{}".format(lex,pos))


# In[15]:


from konlpy.tag import Komoran
kom=Komoran()
result2=kom.pos(string)
for lex, pos in result2:
    print("{}\t{}".format(lex,pos))


# In[16]:


from konlpy.tag import Hannanum
han=Hannanum()
result3=han.pos(string)
for lex, pos in result3:
    print("{}\t{}".format(lex,pos))


# In[17]:


from konlpy.tag import Okt
okt=Okt()
result4=okt.pos(string)
for lex, pos in result4:
    print("{}\t{}".format(lex,pos))


# 띄어쓰기가 제대로 되지 않은 문장

# In[20]:


text=input("띄어쓰기가 제대로 되지 않은 문장을 입력하세요:")
tagged=okt.pos(text)
corrected = ""
for i in tagged:
    if i[1] in ('Josa', 'PreEomi', 'Eomi', 'Suffix', 'Punctuation'):
        corrected += i[0]
    else:
        corrected += " "+i[0]
if corrected[0] == " ":
    corrected = corrected[1:]
print(corrected)


# 불완전한 문장

# In[ ]:


nltk.download('stopwords')


# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[ ]:


nltk.download('stopwords')
text2=input("불완전한 문장을 입력하세요(영어):")
stop_words=set(stopwords.words('english'))
word_tokens=word_tokenize(text2)
result=[]

for w in word_tokens:
    if w not in stop_words:
        result.append(w)
        
print(result)


# In[ ]:





# 속도비교

# In[ ]:




