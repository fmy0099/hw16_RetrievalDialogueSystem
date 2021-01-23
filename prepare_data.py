#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

parser = argparse.ArgumentParser(description='TextRNN text classifier')

parser.add_argument('-in-train-path', type=str, default='data/train.txt', help='训练数据路径')
parser.add_argument('-out-train-path', type=str, default='data/train_split.txt', help='训练数据分词后的路径')
parser.add_argument('-train-file-count', type=int, default=39, help='切分后的训练数据文件数目')
parser.add_argument('-valid-file-count', type=int, default=1, help='切分后的验证数据文件数目')

args = parser.parse_known_args()[0]
# args = parser.parse_args()


# In[ ]:


'''
文档分词
'''
def split_train_file(in_train_path,out_train_path):
    print('split_train_file')
    out = open(out_train_path, 'w',encoding='utf-8') 
    line_no = 0
    with open(in_train_path, 'r',encoding='utf-8') as file:
        line = file.readline()
        while line:
            words = tokenizer(line)
            line_new = ''
            for w in words:
                line_new += w + ' '
            line_new += '\n'
            if len(line_new.strip())==0 and len(line.strip())>0:
                line_new = 'pad\n'
            out.write(line_new)
            line = file.readline()
            if line_no%1000==0:
                sys.stdout.write('\rsplit_train_file line_no = [{}]'.format(line_no))
            line_no += 1
    out.close()
    print('split_train_file finished')


# In[ ]:


import json

'''
加载停用词
'''
def get_stop_words():
    file = open('data/stopwords-iso.json', 'r',encoding='utf-8') 
    stop_words = json.loads(file.read())['zh']   
    file.close() 
    return stop_words


# In[ ]:


import jieba
import re

'''
分词
'''
def tokenizer(text): 
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = regex.sub(' ', text)
    text = text.strip()    
    return [word.strip() for word in jieba.cut(text)]


# In[ ]:


'''
过滤停用词
'''
def stop_filter(words,stop_words):
    ret = []
    for word in words:
        if len(word.strip())>0 and (word not in stop_words):
            ret.append(word)
    return ret   


# In[ ]:


'''
拼接句子
'''
def make_line(words):
    ret = ''
    for word in words:
        ret += word.strip() + ' '
    if len(ret.strip())==0:
        ret = 'pad'
    return ret + '\n'


# In[ ]:


'''
构建词汇表
'''
def make_vocab(in_train_path,stop_words):
    vocab = {}
    line_no = 0
    
    with open(in_train_path, 'r',encoding='utf-8') as file:
        line = file.readline()
        while line:
            words = [word.strip() for word in line.split()]
            words = stop_filter(words,stop_words)
            for w in words:
                if w in vocab.keys():
                    vocab[w] = vocab[w]+1
                else:
                    vocab[w] = 1
            if line_no%1000==0:
                sys.stdout.write('\rline_no = [{}]'.format(line_no))
            line = file.readline()
            line_no+=1 
     
    vocab = dict(sorted(vocab.items(), key = lambda kv:(kv[1], kv[0]),reverse=True))
    print('vocab size = {}'.format(len(vocab)))
          
    stoi = {}
    vocab_list = list(vocab.keys())
    for i in range(len(vocab_list)):
        w = vocab_list[i]
        stoi[w] = i+2 #unk pad
           
    vocab_file = open('data/vocab.txt','w',encoding='utf-8')
    txt = 'name,freq\n'+'unk,10000000\n'+'pad,10000000\n'
    
    for w in vocab.keys():
        txt += w + ',' + str(vocab[w]) + '\n'
        
    vocab_file.write(txt)
    vocab_file.close()
    
    print('vocab finished ')

    return vocab,stoi


# In[ ]:


import random
import sys
import math

def txt2csv(in_train_path,stop_words):
    session = []
    replys = []
    out_train_path,out_valid_path,path = '','',''
    session_count = 0
    with open(in_train_path, 'r',encoding='utf-8') as file:
        line = file.readline()
        while line:
            if line == '\n': 
                session_count+=1 
                replys.append(reply)
            else:
                reply = line.replace(',','').strip()
            line = file.readline()  
     
    count_per_file = session_count/(train_file_count + valid_file_count)
    print('reply count = {} per count = {} '.format(len(replys),count_per_file))
    session_no = 0
    with open(in_train_path, 'r',encoding='utf-8') as file:
        line = file.readline()
        while line:
            if line == '\n':         
                context = ''
                for i in range(len(session)-2):
                    context += session[i]
                
                context = make_line(stop_filter(context.split(),stop_words)).strip()
                query = make_line(stop_filter(session[-2].split(),stop_words)).strip()
                reply = make_line(stop_filter(session[-1].split(),stop_words)).strip()
                neg_reply1 = make_line(stop_filter(replys[random.randint(0,len(replys)-1)].split(),stop_words)).strip()
    
                session_no += 1
                file_no = math.floor(session_no/count_per_file) + 1
                if file_no <= train_file_count:
                    path = 'data/train/train{}.csv'.format(file_no)
                    if path != out_train_path:
                        out_train_path = path
                        out_train_file = open(out_train_path,'w',encoding='utf-8')
                        out_train_file.write('label,context+query,reply\n')                                               
                        out_file = out_train_file
                else:
                    path = 'data/train/valid{}.csv'.format(file_no - train_file_count)
                    if path != out_valid_path:
                        out_valid_path = path
                        out_valid_file = open(out_valid_path,'w',encoding='utf-8')
                        out_valid_file.write('label,context+query,reply\n')                                             
                        out_file = out_valid_file
                    
                out_file.write('{},{}，{},{}\n'.format(1,context,query,reply))
                out_file.write('{},{}，{},{}\n'.format(0,context,query,neg_reply1))
                
                if session_no%500==0:
                    sys.stdout.write('\rpath={} session_no={}/{}\r'.format(path,session_no,session_count))
                
                session = []
            else:
                session.append(line.replace(',','').strip())   
            line = file.readline() 
    out_valid_file.close()
    out_train_file.close()


# In[ ]:


def main(args):
    #加载停用词
    stop_words = get_stop_words()
#     stop_words = []

    #切分train.txt，39个训练数据文件 + 1个验证数据文件
    split_train_file(args.in_train_path,args.out_train_path)
    
    #生成词汇表
    vocab,stoi = make_vocab(args.out_train_path,stop_words)
    
    #训练数据集和验证数据集分词后转成csv文件
    txt2csv(args.out_train_path,stop_words)
    
    #测试数据集分词
    split_train_file('data/seq_replies.txt','data/seq_replies_split.txt')
    split_train_file('data/seq_context.txt','data/seq_context_split.txt')


# In[ ]:


main(args)


# In[ ]:




