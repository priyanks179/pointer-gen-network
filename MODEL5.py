
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#a=np.random.randint(0,50800,(8,100))
#b=np.random.randint(0,50800,(8,25))
#src=torch.tensor(a).long()
#trg=torch.tensor(b).long()
#


rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
eps=1e-12

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-rand_unif_init_mag,rand_unif_init_mag)

class encoder(nn.Module):
    def __init__(self,vocab_size,emb_size,d_model):
        super(encoder,self).__init__()
        
        self.emb=nn.Embedding(vocab_size,emb_size,padding_idx=0)
        self.rnn=nn.LSTM(emb_size,d_model,batch_first=True,bidirectional=True)
        
        init_wt_normal(self.emb.weight)
        init_lstm_wt(self.rnn)        
    def forward(self,x):
        x=self.emb(x)
        x,h=self.rnn(x)
        return x,h#batch,seq_size,2*d_model || (2,batch,d_model)*2
    
class decoder(nn.Module):
    def __init__(self,emb_size,d_model,enc_obj):
        super(decoder,self).__init__()
        self.emb=enc_obj.emb
        self.d_model=d_model
        self.rnn=nn.LSTM(emb_size,d_model,batch_first=True)
        
        self.reduce_cell=nn.Linear(2*d_model,d_model)
        self.reduce_hid=nn.Linear(2*d_model,d_model)
        self.to_emb=nn.Linear(2*d_model+emb_size,emb_size)
        
        init_lstm_wt(self.rnn)
        init_linear_wt(self.reduce_hid)
        init_linear_wt(self.reduce_cell)

    def init_hid(self,enc_hid):
        
        batch=enc_hid[0].shape[1]
        a,b=enc_hid[0].view(1,-1,2*self.d_model),enc_hid[1].view(1,-1,2*self.d_model)
        a.shape
        a,b=enc_hid[0].transpose(0, 1).contiguous().view(batch,-1),enc_hid[1].transpose(0, 1).contiguous().view(batch,-1)      
        a,b=self.reduce_cell(a),self.reduce_hid(b)#1,batch,d_model
        a,b=a.unsqueeze(0),b.unsqueeze(0)
        a,b=F.relu(a),F.relu(b)
        return (a,b)
    def forward (self,x,hid,context):#context --> batch,2*d_model
        x=self.emb(x)
        x1=torch.cat((context,x.squeeze(1)),-1)#batch,2*d_model+emb_size
        x1=self.to_emb(x1).unsqueeze(1)
        op,hid=self.rnn(x1,hid)
        return op,hid,x#batch,1,d_model || (1,batch,d_model)*2
    

class generate_prob(nn.Module):
    def __init__(self,emb_size,d_model):
        super(generate_prob,self).__init__()
        
        self.get_pgen=nn.Linear(4*d_model+emb_size,1)

        init_linear_wt(self.get_pgen)
        
    def forward(self,context,dec_op,dec_inp):
        
        temp=torch.cat((context,dec_op,dec_inp),1)
        prob=torch.sigmoid(self.get_pgen(temp))
        return prob

class generate(nn.Module):
    def __init__(self,d_model,p_gen,max_art_len,vocab_size,oov_size):
        super(generate,self).__init__()
        self.d_model=d_model
        self.oov_size=oov_size
        self.prob_gen=p_gen

        self.mod_enc_op=nn.Linear(2*d_model,2*d_model,bias=False)
        self.mod_dec_op=nn.Linear(2*d_model,2*d_model)
        self.get_attn=nn.Linear(2*d_model,1, bias=False)
        self.get_vocab=nn.Sequential(nn.Linear(3*self.d_model,self.d_model),
                                     nn.Linear(self.d_model,vocab_size-self.oov_size))
        
        self.normalize = nn.BatchNorm1d(2*d_model)

        init_linear_wt(self.mod_dec_op)
        init_linear_wt(self.mod_enc_op)
        init_linear_wt(self.get_attn)
        init_linear_wt(self.get_vocab[0])
        init_linear_wt(self.get_vocab[1])
                
        self.drop=nn.Dropout(p=0.3)
    def forward(self,dec_output,dec_hid,dec_inp,enc_op,enc_op1,src_inp,device):
#        self=model.gen
        
        batch=src_inp.shape[0]
             
        dec_op=torch.cat((dec_hid[0].transpose(0,1),dec_hid[1].transpose(0,1)),1).view(batch,-1)#batch,1,2*d_model
        
        dec_op1=self.mod_dec_op(dec_op)#batch,1,d_model  
        dec_op1=dec_op1.unsqueeze(1).repeat(1,enc_op.shape[1],1)#batch,seq,d_model    
      
        temp1=enc_op1+dec_op1
        features=torch.tanh(temp1)#batch,seq,d_model  
        
        features=self.drop(features)
        
        attn=F.softmax(self.get_attn(features).squeeze(-1),-1)#batch,seq
        attn=self.clean_attn(attn,src_inp)
        
        normalization_factor = attn.sum(1, keepdim=True)
        attn = attn / normalization_factor

        context_vec=torch.bmm(attn.unsqueeze(1),enc_op).squeeze(1)#8,200
        
        context_vec=self.normalize(context_vec)
        
        temp=torch.cat((dec_output.squeeze(1),context_vec),1)  
        
        temp=self.drop(temp)
        
        p_vocab=F.softmax(self.get_vocab(temp),-1)#8,50K
        extra=torch.zeros((enc_op.size(0),self.oov_size)).to(device)#8,800
        p_vocab=torch.cat((p_vocab,extra),dim=-1)#8,50800
        
        p_gen=self.prob_gen(context_vec,dec_op.squeeze(1),dec_inp.squeeze(1))
        
        #######################################################################
        src_attn=torch.mul((1-p_gen),attn)#8,200
        p_final=torch.mul(p_gen,p_vocab)
        p_final=p_final.scatter_add(1, src_inp, src_attn)
        #######################################################################
        temp=(p_final==0).to(torch.float)
        temp=temp*torch.tensor(1e-12).to(device)       
        p_final+=temp        
        return p_final,attn,context_vec

    def clean_attn(self,attn,src):
        mask=(src!=0).float()
        attn=attn*mask
        return attn
    
    def cache_enc_operations(self,enc_op):
#        enc_op=self.reduce_enc_op(enc_op)#batch,seq,d_model
        enc_op1=self.mod_enc_op(enc_op)#batch,seq,d_model
        return enc_op,enc_op1            
    
    def update_context(self,context_1,dec_hid,enc_op1,enc_op,src_inp):
            batch=context_1.shape[0]
             
            dec_op=torch.cat((dec_hid[0].transpose(0,1),dec_hid[1].transpose(0,1)),1).view(batch,-1)#batch,1,2*d_model
        
            dec_op1=self.mod_dec_op(dec_op)#batch,1,d_model  
            dec_op1=dec_op1.unsqueeze(1).repeat(1,enc_op1.shape[1],1)#batch,seq,d_model    
      
            temp1=enc_op1+dec_op1
            features=torch.tanh(temp1)#batch,seq,d_model  
            
            features=self.drop(features)
            
            attn=F.softmax(self.get_attn(features).squeeze(-1),-1)#batch,seq
            attn=self.clean_attn(attn,src_inp)
        
            normalization_factor = attn.sum(1, keepdim=True)
            attn = attn / normalization_factor

            context_vec=torch.bmm(attn.unsqueeze(1),enc_op).squeeze(1)#8,200    
            
            context_vec=self.normalize(context_vec)
            
            return context_vec,attn
    

class Summarize(nn.Module):
    def __init__(self,encoder,decoder,generate,max_art_len,d_model):
        super(Summarize,self).__init__()
        self.enc=encoder
        self.dec=decoder
        self.gen=generate        
        self.max_art_len=max_art_len
        self.d_model=d_model
    def forward(self,src,trg,device):
#        self=model
        
        eps=1e-12
        self.train()
        batch=src.shape[0]
        enc_op,enc_hid=self.enc(src)
        dec_hid=self.dec.init_hid(enc_hid)
        context_1=torch.zeros(batch,2*self.d_model).to(device)
        enc_op,enc_op1=self.gen.cache_enc_operations(enc_op)
        prob=0
        i=0
        step_losses=[]
        context_1,_=self.gen.update_context(context_1,dec_hid,enc_op1,enc_op,src)
        for i in range(trg.shape[1]-1):
            
            
            dec_op,dec_hid,dec_inp=self.dec(trg[:,i].view(-1,1),dec_hid,context_1)
            pred,attn,context_1=self.gen(dec_op,dec_hid,dec_inp,enc_op,enc_op1,src,device)

            tgt=trg[:,i+1].unsqueeze(1)
            gold_probs = torch.gather(pred, 1, tgt).squeeze() 
            
            step_loss = -torch.log(gold_probs+eps)
            
            
            dec_mask=(tgt!=0).view(-1).float()
            step_loss=step_loss*dec_mask
            step_losses.append(step_loss)
                        
            if i==0:
                prob=pred.unsqueeze(1)
            else:
                prob=torch.cat((prob,pred.unsqueeze(1)),1)
            
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/(trg.shape[1]-1)
        loss = torch.mean(batch_avg_loss)            
                
        return prob,loss
      
    def evaluate(self,src,trg,trg_len=26,device='cuda'):
        
        eps=1e-12
        batch=src.shape[0]
        enc_op,enc_hid=self.enc(src)
        dec_hid=self.dec.init_hid(enc_hid)
        context_1=torch.zeros(batch,2*self.d_model).to(device)
        dec_inp=(torch.ones(batch,1).long()*2).to(device)
        enc_op,enc_op1=self.gen.cache_enc_operations(enc_op)
        prob=0    
        step_losses=[]
        context_1,_=self.gen.update_context(context_1,dec_hid,enc_op1,enc_op,src)
        for i in range(trg_len-1):
            dec_op,dec_hid,dec_inp=self.dec(dec_inp,dec_hid,context_1)
            pred,attn,context_1=self.gen(dec_op,dec_hid,dec_inp,enc_op,enc_op1,src,device)

            tgt=trg[:,i+1].unsqueeze(1)
            gold_probs = torch.gather(pred, 1, tgt).squeeze() 
            step_loss = -torch.log(gold_probs+eps)
            
            dec_mask=(tgt!=0).view(-1).float()
            step_loss=step_loss*dec_mask
            
            step_losses.append(step_loss)  
            
            if i==0:
                prob=pred.unsqueeze(1)
            else:
                prob=torch.cat((prob,pred.unsqueeze(1)),1)
            dec_inp=pred.argmax(-1).view(-1,1)
            
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/(trg.shape[1]-1)
        loss = torch.mean(batch_avg_loss)              
        return prob,loss           
            
        
def initialise_model(emb_size=64,d_model=128,vocab_size=50800,oov_size=800,max_art_len=1000):
    enc=encoder(vocab_size,emb_size,d_model)
    dec=decoder(emb_size,d_model,enc)
    gen=generate(d_model,generate_prob(emb_size,d_model),max_art_len,vocab_size,oov_size)
    summ=Summarize(enc,dec,gen,max_art_len,d_model)
    return summ

#model=initialise_model()
#emb_size=32
#d_model=64
#vocab_size=50800
#oov_size=800
#max_art_len=1000
#device='cpu'
#src_inp=src