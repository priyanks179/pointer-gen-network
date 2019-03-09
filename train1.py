import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from packages.vocab import Vocab
from packages.functions import num_to_var
import util
import eval1
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import rouge
import MODEL5 as MODEL
device='cuda'

r = rouge.Rouge(metrics=[ 'rouge-2'])

#dir='data/train_32_extracted_rnn'
dir='data1/train_32'

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def get_filelst(dir):
    for _,_,d in os.walk(dir):
        names=d
    for ind,val in enumerate(names):
        names[ind]=dir+'/'+val
    return names
names=get_filelst(dir)
a=[]
for x in sorted_nicely(names):
    a.append(x)
names=a[:]
del a

def get_token(tensor,pad=0):
    return (tensor[:,1:] != pad).data.sum()
def update(x,y,tokens,loss):
    
#    X=x.contiguous().view(-1,x.size(-1))
#    X=torch.log(X)
#    Y=y.contiguous().view(-1)
#    Y=Variable(Y, requires_grad=False)
#    loss=criteria(X,Y)
#    
#    
#    loss/=torch.tensor(tokens).to(torch.float)
    optim.zero_grad()
#    start=time.time()
    loss.backward()
#    print(time.time()-start)
    
    torch.nn.utils.clip_grad_norm(model.parameters(),2.0)
    optim.step()
    loss=loss.detach()
    
#    del X,Y
    return loss

model=MODEL.initialise_model().to(device)
criteria=torch.nn.NLLLoss(ignore_index=0,reduction='sum')
optim=torch.optim.Adagrad(model.parameters(),lr=0.15,initial_accumulator_value=0.1)
#optim=torch.optim.Adam(model.parameters())
if os.path.isfile('pntr_attn'):
    model.load_state_dict(torch.load('pntr_attn', map_location=lambda storage, loc: storage))
    optim_state = torch.load('optim')
    optim.load_state_dict(optim_state)

with open('stats/count.txt','r') as handle :
        count_lst=handle.read().strip().split('\n') 
 
   
iter=0
batch_size=8

count=len(count_lst)%len(names)
eval=eval1.Infer()
for iter in range(1800003):
#    break
    if count>=len(names):
        count=0
    if iter%(32/batch_size)==0:
        file=names[count] 
        input,target,idx2oov=util.get_id(file,input_len=100,target_len=25)
        ind=0        
        with open('stats/count.txt','a') as handle :
            handle.write(str(count)+'\n') 
        count+=1
        l=[j for j in range(int(32/batch_size))]
    inp=num_to_var(input[ind*batch_size:ind*batch_size+batch_size,:]).to(device)#8,400
    decoder_inp=num_to_var(target[ind*batch_size:ind*batch_size+batch_size,:]).to(device)        
    
    
    prob,loss=model(inp,decoder_inp,device)
    loss=update(prob,decoder_inp[:,1:],get_token(decoder_inp[:,1:]),loss)

    if iter%50==0:
        name=file.split('/')[2].split('.')[0]
        pred=prob.argmax(-1)
        real_summ=util.list_to_summ(decoder_inp[-1,1:].cpu().numpy().tolist(),idx2oov)
        summ=util.list_to_summ(pred[-1,:].cpu().numpy().tolist(),idx2oov)
        e_real_summ,e_summ,e_loss,e_name=eval.infer(model)
        score = r.get_scores(summ, real_summ)[0]['rouge-2']['f']
        e_score=r.get_scores(e_summ, e_real_summ)[0]['rouge-2']['f']
        print('iter: ',iter,' loss: ', loss.item(),' e_loss: ',e_loss,' rouge: ',score,' e_rouge: ',e_score,)
        print('pred summ : ',summ,'\n------------------------------') 
        print('real summ : ',real_summ,'\n--------------------------\n')
        print('eval summ : ',e_summ,'\n------------------------------') 
        print('orig summ : ',e_real_summ,'\n--------------------------')
        print('file: ',name,' e_file: ',e_name,'\n')
        
    
        
    
        with open('stats/loss.txt','a') as handle :
            handle.write(str(loss.item())+'\n')    
    if iter%1000==0  and iter!=0:
        torch.save(model.state_dict(),'pntr_attn')        
        torch.save(optim.state_dict(),'optim')

if os.path.isfile('stats/loss.txt')==1:
    with open('stats/loss.txt', 'r') as output:
        lost_list=output.read().strip().split('\n')    
        for i,data in enumerate(lost_list):
            try:
                a=float(data)
                if a!=0 :
                  lost_list[i]=a 
            except:
                pass 

if os.path.isfile('stats/e_loss.txt')==1:
    with open('stats/e_loss.txt', 'r') as output:
        e_loss=output.read().strip().split('\n')    
        for i,data in enumerate(e_loss):
            try:
                a=float(data)
                if a!=0 :
                  e_loss[i]=a  
            except:
                pass 
            
temp=[]
for i in range(0,len(lost_list)):
    try:
        if lost_list[i]<10:
            temp.append(lost_list[i])
    except:
        pass
lost_list=temp

temp=[]
for i in range(0,len(e_loss)):
    try:
        if e_loss[i]<10:
            temp.append(e_loss[i])
    except:
        pass
e_loss=temp   

plt.plot(lost_list,'r',label='train_loss')
plt.plot(e_loss,'g',label='eval_loss')
plt.legend()
plt.show()

#bar=[i/2 for i in range(0,20)]
#plt.hist(e_loss[4000:],bar,histtype='bar',rwidth=0.8)

