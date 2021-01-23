#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import argparse
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchtext import data
import re
from torchtext.vocab import Vectors
import numpy as np
from torch.autograd import Variable
import json
import threading

parser = argparse.ArgumentParser(description='RNN QA')

parser.add_argument('-lr', type=float, default=0.01, help='学习率')
parser.add_argument('-batch-size', type=int, default=128)
parser.add_argument('-context-len', type=int, default=200)
parser.add_argument('-epoch', type=int, default=1)
parser.add_argument('-embedding-dim', type=int, default=300, help='词向量的维度')
parser.add_argument('-hidden_size', type=int, default=128, help='lstm中神经单元数')
parser.add_argument('-layer-num', type=int, default=1, help='lstm stack的层数')
parser.add_argument('-bidirectional', type=bool, default=True, help='是否使用双向lstm')
parser.add_argument('-static', type=bool, default=True, help='是否使用预训练词向量')
parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=100, help='经过多少iteration对验证集进行测试')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')
parser.add_argument('-vocab-path', type=str, 
                    default='D:/Summer/DeepL-data/glove.840B.300d/glove.840B.300d.txt', 
                    help='词向量,static为True时生效')
args = parser.parse_args()


# In[ ]:


'''
停用词
'''
def get_stop_words():
    file = open('data/stopwords-iso.json', 'r',encoding='utf-8') 
    stop_words = json.loads(file.read())['zh']   
    file.close() 
    return stop_words


# In[ ]:


'''
分词
'''
def tokenizer(text): 
    all_words = [word for word in text.split()]
    return all_words


# In[ ]:


'''
加载词汇表
'''
def load_vocab(args):
    print('Load vocab...')
#     text = data.Field(sequential=True,lower=True, stop_words=args.stop_words, tokenize=tokenizer)
    text = data.Field(sequential=False,lower=True,tokenize=tokenizer)
    label = data.Field(sequential=True)
    
    txt_vocab = data.TabularDataset.splits(
            path='data/',
            skip_header=True,
            train='vocab.txt',
            format='csv',
            fields=[('name',text),('freq',label)],
        )[0]
#     text.build_vocab(txt_vocab,max_size=420000,vectors=Vectors(name = args.vocab_path)) 
    text.build_vocab(txt_vocab,vectors=Vectors(name = args.vocab_path))
    
    vocab_iter = data.Iterator(
            txt_vocab,
            sort_key=lambda x: len(x.label),
            batch_size=len(txt_vocab), 
            device=-1
    )
    
    args.embedding_dim = text.vocab.vectors.size()[-1]
    args.vectors = text.vocab.vectors    
    args.stoi = vars(text.vocab)['stoi']    
    args.itos = vars(text.vocab)['itos']    
    args.vocab_size = len(text.vocab)   
    args.label_num = 2
    
#     print(list(args.stoi)[:10])
    
    print('vocab size : ',args.vocab_size)


# In[ ]:


'''
当前训练数据文件的词，映射到词汇表中
'''
def cvt_dict(args,batch):
    for i in range(batch.size(0)):
        for j in range(batch.size(1)):
            index = batch[i][j]
            name = args.cur_itos[index]
            if name in args.stoi.keys():
                batch[i][j] = args.stoi[name]
            else:
                batch[i][j] = args.stoi[0]
    return batch


# In[ ]:


'''
加载训练数据文件和验证数据文件
'''
def load_data(args):
    print('加载数据:{}'.format(args.train_file))
       
    text = data.Field(sequential=True, fix_length=args.context_len, lower=True, tokenize=tokenizer)
    label = data.Field(sequential=False)
    
    train, val = data.TabularDataset.splits(
            path='data/train/',
            skip_header=True,
            train=args.train_file,
            validation='valid.csv',
            format='csv',
            fields=[('label', label), ('query', text),('reply', text)],
        ) 

    if args.static:
        text.build_vocab(train, val, vectors=Vectors(name = args.vocab_path)) 
        args.cur_itos = vars(text.vocab)['itos']
    else: 
        text.build_vocab(train, val)

    label.build_vocab(train, val)

    train_iter, val_iter = data.Iterator.splits(
            (train, val),
            sort_key=lambda x: len(x.query),
            batch_sizes=(args.batch_size, args.batch_size*2), 
            device=-1
    )
    
    args.label_num = len(label.vocab)-1
    print(vars(train.examples[0]))

    return train_iter, val_iter


# In[ ]:


class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        embedding_dim = args.embedding_dim
        label_num = args.label_num
        vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.bidirectional = args.bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if args.static:  
            v = Vectors(name = args.vocab_path)
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        self.lstm = nn.LSTM(embedding_dim, # x的特征维度,即embedding_dim
                            self.hidden_size,# 隐藏层单元数
                            self.layer_num,# 层数
                            batch_first=True,# 第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
                            bidirectional=self.bidirectional) # 是否用双向
        self.fc = nn.Linear(self.hidden_size * 2 , label_num) 
        
    
    def forward(self, query,reply): 
        x = torch.cat((query,reply), 0)             
        x = self.embedding(x)      
        h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(self.layer_num, x.size(0),self.hidden_size)
        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(self.layer_num, x.size(0),self.hidden_size)       
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out1,out2 = torch.split(out, [query.size(0),reply.size(0)], dim=0)          
        out1 = torch.einsum('bik->bki',out1)
        out1 = torch.einsum('bik,kj->bij',out1,self.w1)
        out3 = torch.einsum('bki,bij->bj', out1,out2)
        out = self.fc(out3[:, :])
        return out


# In[ ]:


def train(args,train_iter, dev_iter):
    steps = 0
    model.train()
    for epoch in range(1, args.epoch + 1):            
        for batch in train_iter:
            f1,f2, target = batch.query,batch.reply,batch.label
           
            with torch.no_grad():
                f1.t_()
                f2.t_()
                target.sub_(1)
     
            optimizer.zero_grad()           
            logits = model(cvt_dict(args,f1),cvt_dict(args,f2))            
            model.loss = F.cross_entropy(logits, target)
            model.loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1] == target).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rEpoch[{}] Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,steps,
                                                                             model.loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > model.best_acc:
                    model.best_acc = dev_acc
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(model.best_acc))
                        save(model,optimizer,args.save_dir, 'best', steps)

    print('\rtrain finish')


# In[ ]:


'''
验证、测试
'''
def eval(data_iter, model, args):
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        f1,f2, target = batch.query,batch.reply,batch.label          
        with torch.no_grad():
            f1.t_()
            f2.t_()
            target.sub_(1)
        logits = model(f1,f2)
        pre = torch.max(logits, 1)[1]
        
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()) == target).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


# In[ ]:


def save(model,optimizer,save_dir,save_prefix,steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    
    save_path = '{}.pt'.format(save_prefix)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': model.loss,
            'w1':model.w1,
            'best_acc':model.best_acc,
            }, save_path)


# In[ ]:


def loadModel(args):
    model = TextRNN(args)
    model.best_acc = 0
    dtype = torch.FloatTensor
#     model.w1 = Variable(torch.randn(args.hidden_size * 2, args.hidden_size * 2).type(dtype), requires_grad=True) if args.bidirectional else Variable(torch.randn(args.hidden_size, 100).type(dtype), requires_grad=True)
    model.w1 = Variable(torch.randn(args.context_len,args.context_len).type(dtype), requires_grad=True) if args.bidirectional else Variable(torch.randn(args.hidden_size, 100).type(dtype), requires_grad=True)
    model.epoch = 0
    model.loss = 0
    
#     if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    save_path = os.path.join(args.save_dir, 'best.pt')   
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])        
        model.loss = checkpoint['loss']
        model.w1 = checkpoint['w1']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.best_acc = checkpoint['best_acc']
        model.eval()
        print('load exist model',save_path,'best acc:',model.best_acc)  
  
    return model,optimizer


# In[ ]:


def train_loop():
    for k in range(1,40):
        args.train_file = 'train{}.csv'.format(k)
        train_iter, dev_iter = load_data(args) 
        print('加载数据完成:{}'.format(args.train_file))
        train(args,train_iter,dev_iter)


# In[ ]:


'''
预测
'''
def predict_model(model,query,replys):
    predict = []
    
    for i in range(len(replys)): 
        f1 = query
        f2 = replys[i].unsqueeze(dim=0)
        logits = model(f1,f2)
        pre = logits[0][1]
        predict.append(pre)

    m = torch.max(torch.tensor(predict))
    for k in range(len(predict)):
        if predict[k]==m:
            return k


# In[45]:


def build_vocab(ones,fixlen):
    batch = []
    for one in ones:
        t = []
        words = one.split()
        for w in words:
            if w in args.stoi.keys():
                t.append(args.stoi[w])
            else:
                t.append(args.stoi[0])
        batch.append(t[:fixlen])
    
    for one in batch:
        for k in range(len(one),fixlen):
            one.append(0)
            
    return torch.tensor(batch)


# In[ ]:


def predict_test():
    query = '是 我们 专业 的 宿舍 啊  你 是 我们 班 的 吗 曾经 是 有种 好 伤感 的 感觉 '
    replys = ('我 中午 哪肿 么 啦',
                           '是 的',
                           '啊 偶 抚摸',
                              '谢谢 啦', 
    '棒棒 的 你',
    '哈哈  我 一直 在 啊', 
    '我 没有 在 长宁  打钱 嘛  手机 版 的', 
    '嗯 嗯', 
    '嗯 啊', 
    '是 的 呀  这 周末 回去 ')
    f1 = build_vocab([query],args.context_len)    
    f2 = build_vocab(replys,args.context_len)         
    index = predict_model(model,f1,f2)
    print(index,replys[index])


# In[51]:


'''
对分词后的seq_context.txt，seq_replies.txt进行预测,生成结果txt
'''
def do_predict_work():
    context_path = 'data/seq_context_split.txt'
    reply_path = 'data/seq_replies_split.txt'
    out_path = 'data/result.txt'

#     load_vocab(args)
#     model,optimizer = loadModel(args) 

    with open(context_path, 'r',encoding='utf-8') as file:
        tmp = file.read()
    context = tmp.split('\n\n')

    with open(reply_path, 'r',encoding='utf-8') as file:
        tmp = file.read()
    replys = tmp.split('\n\n')
    
    
    result = []

    for i in range(len(context)):
        c = context[i].replace('\n','').strip()
        rs = replys[i].split('\n')
        f1 = build_vocab([c],args.context_len)    
        f2 = build_vocab(rs,args.context_len)         
        index = predict_model(model,f1,f2)
        result.append(index)
        if i%10==0:
            sys.stdout.write('\rpredict {}/{}\r'.format(i,len(context)))
#         print('Q:',i,c)
#         print('A: ',index,rs[index],'\n')
        
    with open(out_path, 'w',encoding='utf-8') as file:
        for idx in result:
            file.write(str(idx)+'\n')


# In[ ]:


#预测测试
# predict_test()


# In[ ]:


#手动修改学习速率
# args.lr=0.05


# In[ ]:


#手动保存模型
# save(model,optimizer,args.save_dir, 'best', 0)


# In[ ]:


#加载词汇表
load_vocab(args)
#加载模型，不存在则新建
model,optimizer = loadModel(args) 
#启动训练线程
t = threading.Thread(target=train_loop,args=[])
t.start() 


# In[ ]:


#对分词后的seq_context.txt，seq_replies.txt进行预测,生成结果txt
do_predict_work()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




