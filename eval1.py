import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from packages.vocab import Vocab
from packages.functions import num_to_var
import util


def get_filelst(dir):
    for _,_,d in os.walk(dir):
        names=d
    for ind,val in enumerate(names):
        names[ind]=dir+'/'+val
    return names


def get_token(tensor,pad=0):
    return (tensor[:,1:] != pad).data.sum()
def calc_loss(x,y,criteria):
    X=x.contiguous().view(-1,x.size(-1))
    X=torch.log(X)
    tokens=get_token(y[:,1:])
    Y=y.contiguous().view(-1)
    Y=Variable(Y, requires_grad=False)
    loss=criteria(X,Y)
    loss=loss.detach()
    loss=loss/torch.tensor(tokens).to(torch.float)
    del X,Y
    return loss

class Infer:
    def __init__(self,device='cuda',max_len=100):
        self.device=device
        self.count=0
        self.ind=0
#        self.dir='data/val_32_extracted_rnn'
        self.dir='data1/val_32'
#        self.file_list = os.listdir(path='data/val_32_extracted_rnn')
        self.file_list = os.listdir(path='data1/val_32')
        
        self.batch_size=8
        self.input,self.target,self.idx2oov=None,None,None
        self.sos=2
        self.max_len,self.summ_len=max_len,25
        self.iter=0
        self.optim=torch.nn.NLLLoss(ignore_index=0,reduction='sum')
        self.file=0
    def infer(self,model):
#        self=eval

        model.eval()
        if self.count>=len(self.file_list):
            self.count=0
        if self.iter%(32/self.batch_size)==0:
            self.file=self.file_list[self.count] 
            path=os.path.join(self.dir,self.file)
            self.input,self.target,self.idx2oov=util.get_id(path,input_len=self.max_len,target_len=self.summ_len)
            self.ind=0
            self.count+=1
            with open('stats/e_count.txt','a') as handle :
                handle.write(str(self.count)+'\n')   
        with torch.no_grad():
            src=num_to_var(self.input[self.ind*self.batch_size:self.ind*self.batch_size+self.batch_size,:]).to(self.device)#8,400
            decoder_inp=num_to_var(self.target[self.ind*self.batch_size:self.ind*self.batch_size+self.batch_size,:]).to(self.device)
        self.ind+=1  

        prob,loss=model.evaluate(src,decoder_inp,trg_len=self.summ_len+1)
#        loss=calc_loss(prob,decoder_inp[:,1:],self.optim)
        pred=prob.argmax(-1)
        
        real_summ=util.list_to_summ(decoder_inp[-1,1:].cpu().numpy().tolist(),self.idx2oov)
        summ=util.list_to_summ(pred[-1,:].cpu().numpy().tolist(),self.idx2oov)
        
        with open('stats/e_loss.txt','a') as handle :
            handle.write(str(loss.item())+'\n')        
            
        self.iter+=1
        
        return real_summ,summ,loss.item(),self.file

