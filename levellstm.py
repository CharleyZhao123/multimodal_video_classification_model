# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnet import resnet18, resnet50, resnet101
from core import model

torch.manual_seed(1)

class Flstm(nn.Module):

    def __init__(self, feature_dim, nclass,hidden_dim,num_dim,outnet_name='resnet18',innet_name='mobilefacenet'):
        super(Flstm, self).__init__()

        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim   # [out,in]
        self.rnn_out = nn.GRU(self.feature_dim[0], self.hidden_dim[0],1,batch_first=True)
        self.rnn_in = nn.GRU(self.feature_dim[1], self.hidden_dim[1],1,batch_first=True)
        self.rnn_num = nn.GRU(self.feature_dim[2], self.hidden_dim[2],1,batch_first=True)
    
        # 将LSTM的输出映射到类别
        self.hidden2tag = nn.Linear(sum(self.hidden_dim), nclass)  
        # self.in2tag = nn.Linear(self.hidden_dim[1],nclass)
        # self.out2tag = nn.Linear(self.hidden_dim[0],nclass) 
        
        #self.transfc = nn.Linear(in_features=self.hidden_dim[0],out_features=self.hidden_dim[1]) 
        # 对原始图像/数值数据的各特征提取模型
        if outnet_name=='resnet18':
            outnet = resnet18(pretrained=True)
        elif outnet_name=='resnet50':
            outnet = resnet50(pretrained=True)
        elif outnet_name=='resnet101':
            outnet = resnet101(pretrained=True)
            
        if innet_name=='resnet18':
            innet = resnet18(pretrained=True)
        elif innet_name=='mobilefacenet':
            innet = model.MobileFacenet()
            innet_dict = innet.state_dict()
            resume = './core/068.ckpt'
            ckpt = torch.load(resume)
            pretrained_dict = {k: v for k, v in ckpt['net_state_dict'].items() if k in innet_dict}
            innet_dict.update(pretrained_dict)
            innet.load_state_dict(innet_dict)

        numnet = nn.Linear(num_dim,self.feature_dim[2])
        
        self.outnet = outnet.cuda()
        self.innet = innet.cuda()
        self.numnet = numnet.cuda()
        
    def forward(self,data):
        in_data = data[0]
        out_data = data[1]
        num_data = data[2]
        batch_size = in_data.shape[0]
        seq_len = in_data.shape[1]
        in_data=in_data.view(batch_size*seq_len,in_data.shape[2],in_data.shape[3],in_data.shape[4]) 
        in_data=in_data.cuda()
        out_data=out_data.view(batch_size*seq_len,out_data.shape[2],out_data.shape[3],out_data.shape[4]) 
        out_data=out_data.cuda()
        num_data = num_data.view(batch_size*seq_len,num_data.shape[2])
        num_data = num_data.cuda()
        # 送入特征提取网络
        infeature = self.innet(in_data)
        outfeature = self.outnet(out_data)
        numfeature = self.numnet(num_data)
        infeature =  infeature.view(batch_size,seq_len,infeature.shape[1])
        #infeature = self.transfc(infeature)
        outfeature =  outfeature.view(batch_size,seq_len,outfeature.shape[1])       # b,sl,fo 
        numfeature = numfeature.view(batch_size,seq_len,numfeature.shape[1])
        # 送入时序网络
        _,ho = self.rnn_out(outfeature)
        _,hi = self.rnn_in(infeature)
        _,hn = self.rnn_num(numfeature)
        # 模态融合
        hc = torch.cat((ho,hi,hn),dim=-1)
        res_concat = self.hidden2tag(hc)
        return res_concat