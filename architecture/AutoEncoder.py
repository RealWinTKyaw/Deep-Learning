import torch
import torch.nn.functional as F
import torch.nn as nn

class Encoder(torch.nn.Module):
    def __init__(self,inputs,outputs,kernel):
        super(Encoder, self).__init__()
        
        self.conv=nn.Conv2d(inputs, outputs,kernel_size=kernel)
        self.batchnorm=nn.BatchNorm2d(outputs)
        self.relu=nn.LeakyReLU()
        
    def forward(self,x):
        x=self.conv(x)
        x=self.batchnorm(x)
        x=self.relu(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self,inputs,outputs,kernel):
        super(Decoder,self).__init__()
        self.convt=nn.ConvTranspose2d(inputs,outputs,kernel_size=kernel)
        self.batchnorm=nn.BatchNorm2d(outputs)
        self.relu=nn.ReLU()
        
    def forward(self,x):
        x=self.convt(x)
        x=self.batchnorm(x)
        x=self.relu(x)
        return x
    
class AutoEncoder(torch.nn.Module):
    def __init__(self, hidden, kernel_size):
        super(AutoEncoder,self).__init__()
        self.enc= [Encoder(hidden[i], hidden[i+1], kernel_size) for i in range(len(hidden)-1)]
        self.enc_combined = nn.Sequential(*self.enc)

        self.dec= [Decoder(hidden[i], hidden[i-1], kernel_size) for i in range(len(hidden)-1, 0, -1)]
        self.dec_combined = nn.Sequential(*self.dec)
        
    def forward(self,x):
        x=self.enc_combined(x)
        x=self.dec_combined(x)
        return x
    
    def get_features(self, x):
        return self.enc_combined(x)