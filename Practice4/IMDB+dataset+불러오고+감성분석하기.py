
# coding: utf-8

# In[10]:

import os
files = os.listdir('aclimdb/train/pos')


# In[12]:

first_file = files[0]
with open('aclimdb/train/pos/{}'.format(first_file),'r',encoding='utf-8') as f:
    review = f.read()
    f.close()
    


# In[13]:

review


# In[16]:

#전체 긍정 데이터 저장하기
pos_train_list=[]
for file in files:
    with open('aclimdb/train/pos/{}'.format(file),'r',encoding='utf-8') as f:
        review = f.read()
        f.close()
    pos_train_list.append(review)
print(len(pos_train_list))


# In[17]:

import nltk
from nltk.corpus import sentiwordnet as swn


# In[18]:

swn.senti_synsets('hate')


# In[19]:

list(swn.senti_synsets('hate'))


# In[20]:

list(swn.senti_synsets('hate','v'))


# In[21]:

list(swn.senti_synsets('hate','v'))[0].pos_score()


# In[22]:

list(swn.senti_synsets('hate','v'))[0].neg_score()


# In[27]:

def word_sentiment_calculator(word, tag):
    pos_score =0
    neg_score =0
    
    if 'NN' in tag and len(list(swn.senti_synsets(word, 'n')))>0:
        syn_set = list(swn.senti_synsets(word, 'n'))
    elif 'VB' in tag and len(list(swn.senti_synsets(word, 'v')))>0:
        syn_set = list(swn.senti_synsets(word, 'v'))
    elif 'JJ' in tag and len(list(swn.senti_synsets(word, 'a')))>0:
        syn_set = list(swn.senti_synsets(word, 'a'))
    elif 'RB' in tag and len(list(swn.senti_synsets(word, 'r')))>0:
        syn_set = list(swn.senti_synsets(word, 'r'))
    else :
        return (0,0)
    
    for syn in syn_set:
        pos_score += syn.pos_score()
        neg_score += syn.neg_score()
    return (pos_score/len(syn_set), neg_score/len(syn_set))
        


# In[ ]:




# In[30]:

word_sentiment_calculator('love','NN')


# In[31]:

word_sentiment_calculator('love','VB')


# In[38]:

sent = 'I hate you'
tokens = nltk.word_tokenize(sent)
pos_tags = nltk.pos_tag(tokens)
pos_tags


# In[52]:

def sentence_sentiment_calculator(a):
    tokens = nltk.word_tokenize(a)
    pos_tags = nltk.pos_tag(tokens)
    pos_score = 0
    neg_score =0
    for word, tag in pos_tags:
        pos_score += word_sentiment_calculator(word,tag)[0]
        neg_score += word_sentiment_calculator(word,tag)[1]
    return (pos_score, neg_score)
        


# In[54]:

sentence_sentiment_calculator(review)


# In[ ]:



