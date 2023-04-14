import torch
import torch.nn.functional as F
import torch.nn as nn

class DeepLearnLinear(torch.nn.Module):
    def __init__(self, inputs, outputs, dropout):
        super(DeepLearnLinear, self).__init__()
        
        self.linear = nn.Linear(inputs, outputs)
        nn.init.xavier_uniform_(self.linear.weight)
        self.batch_norm = nn.BatchNorm1d(outputs)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        return x
    
class ConvEmbed(torch.nn.Module):
    def __init__(self, inputs, outputs, kernel_size, dropout):
        super(ConvEmbed, self).__init__()
        
        self.conv = nn.Conv2d(inputs, outputs, kernel_size = kernel_size)
        self.batch_norm = nn.BatchNorm2d(outputs)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        return x
    
class AttentionLayer(nn.Module):
    def __init__(self, inputs):
        super(AttentionLayer, self).__init__()
        
        self.inputs = inputs
        self.query = nn.Linear(inputs, inputs)
        self.key = nn.Linear(inputs, inputs) 
        self.value = nn.Linear(inputs, inputs)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query(x).view(batch_size, -1, self.inputs)
        key = self.key(x).view(batch_size, -1, self.inputs)
        value = self.value(x).view(batch_size, -1, self.inputs)
        
        intermediate = torch.bmm(query, key.transpose(1, 2))/(self.inputs**0.5)
        attention_weights = F.softmax(intermediate, dim = 2)
        output = torch.bmm(attention_weights, value).view(batch_size, -1)
        return output
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, inputs, output, dropout):
        super(TransformerBlock, self).__init__()
        
        self.residuals = inputs
        self.attention = AttentionLayer(inputs)
        self.linear = nn.Linear(inputs, inputs)
        self.batch_norm1 = nn.BatchNorm1d(inputs)
        
        self.highway = nn.Linear(inputs, inputs)
        self.transform = nn.Linear(inputs, inputs)
        self.batch_norm2 = nn.BatchNorm1d(inputs)
        
        self.output = DeepLearnLinear(inputs, output, dropout)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.linear(x)
        x += self.residuals
        x = self.batch_norm1(x)
        save = x.detach().clone()
        
        h = self.highway(x)
        t_gate = torch.sigmoid(self.transform(x))
        c_gate = 1 - t_gate
        x = h*t_gate + x*c_gate
        x += save
        x = self.batch_norm2(x)
        x = self.output(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, hidden, kernel_size, window, blocks, dropout=0.5, labels=2):
        super(Transformer, self).__init__()
        
        self.conv = [ConvEmbed(hidden[i], hidden[i+1], kernel_size, dropout) for i in range(len(hidden)-1)]
        self.conv_combined = nn.Sequential(*self.conv)
        self.maxpool = nn.MaxPool2d(window)
        
        self.flattened = blocks[0]
        self.blocks = [TransformerBlock(blocks[i], blocks[i+1], dropout) for i in range(len(blocks)-1)]
        self.blocks_combined = nn.Sequential(*self.blocks)
        
        self.output = DeepLearnLinear(blocks[-1], labels, dropout)

    def forward(self, x):
        x = self.conv_combined(x)
        x = self.maxpool(x)
        x = x.view(-1, self.flattened)
        x = self.blocks_combined(x)
        x = self.output(x)
        return x