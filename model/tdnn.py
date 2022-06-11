# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# ''' Res2Conv1d + BatchNorm1d + ReLU
# '''
# class Res2Conv1dReluBn(nn.Module):
#     '''
#     in_channels == out_channels == channels
#     '''
#     def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
#         super().__init__()
#         assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
#         self.scale = scale
#         self.width = channels // scale
#         self.nums = scale if scale == 1 else scale - 1

#         self.convs = []
#         self.bns = []
#         for i in range(self.nums):
#             self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
#             self.bns.append(nn.BatchNorm1d(self.width))
#         self.convs = nn.ModuleList(self.convs)
#         self.bns = nn.ModuleList(self.bns)

#     def forward(self, x):
#         out = []
#         spx = torch.split(x, self.width, 1)
#         for i in range(self.nums):
#             if i == 0:
#                 sp = spx[i]
#             else:
#                 sp = sp + spx[i]
#             # Order: conv -> relu -> bn
#             sp = self.convs[i](sp)
#             sp = self.bns[i](F.relu(sp))
#             out.append(sp)
#         if self.scale != 1:
#             out.append(spx[self.nums])
#         out = torch.cat(out, dim=1)
#         return out



# ''' Conv1d + BatchNorm1d + ReLU
# '''
# class Conv1dReluBn(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
#         super().__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
#         self.bn = nn.BatchNorm1d(out_channels)

#     def forward(self, x):
#         return self.bn(F.relu(self.conv(x)))



# ''' The SE connection of 1D case.
# '''
# class SE_Connect(nn.Module):
#     def __init__(self, channels, s=4):
#         super().__init__()
#         assert channels % s == 0, "{} % {} != 0".format(channels, s)
#         self.linear1 = nn.Linear(channels, channels // s)
#         self.linear2 = nn.Linear(channels // s, channels)

#     def forward(self, x):
#         out = x.mean(dim=2)
#         out = F.relu(self.linear1(out))
#         out = torch.sigmoid(self.linear2(out))
#         out = x * out.unsqueeze(2)
#         return out

# ''' SE-Res2Block.
#     Note: residual connection is implemented in the ECAPA_TDNN model, not here.
# '''
# def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
#     return nn.Sequential(
#         Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
#         Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
#         Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
#         SE_Connect(channels)
#     )



# ''' Attentive weighted mean and standard deviation pooling.
# '''
# class AttentiveStatsPool(nn.Module):
#     def __init__(self, in_dim, bottleneck_dim):
#         super().__init__()
#         # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
#         self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1) # equals W and b in the paper
#         self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1) # equals V and k in the paper

#     def forward(self, x):
#         # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
#         alpha = torch.tanh(self.linear1(x))
#         alpha = torch.softmax(self.linear2(alpha), dim=2)
#         mean = torch.sum(alpha * x, dim=2)
#         residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
#         std = torch.sqrt(residuals.clamp(min=1e-9))
#         return torch.cat([mean, std], dim=1)



# ''' Implementation of
#     "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".
#     Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
#     because it brings little improvment but significantly increases model parameters. 
#     As a result, this implementation basically equals the A.2 of Table 2 in the paper.
# '''
# class ECAPA_TDNN(nn.Module):
#     def __init__(self, in_dim=80, hidden_dim=512, embedding_size=192):
#         super().__init__()
#         self.layer1 = Conv1dReluBn(in_dim, hidden_dim, kernel_size=5, padding=2)
#         self.layer2 = SE_Res2Block(hidden_dim, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
#         self.layer3 = SE_Res2Block(hidden_dim, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
#         self.layer4 = SE_Res2Block(hidden_dim, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

#         cat_channels = hidden_dim * 3
#         self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
#         self.pooling = AttentiveStatsPool(cat_channels, 128)
#         self.bn1 = nn.BatchNorm1d(cat_channels*2)
#         self.linear = nn.Linear(cat_channels*2, embedding_size)
#         self.bn2 = nn.BatchNorm1d(embedding_size)

#     def forward(self, x):
# #         x = x.transpose(1, 2)
#         out1 = self.layer1(x)
#         out2 = self.layer2(out1) + out1
#         out3 = self.layer3(out1 + out2) + out1 + out2
#         out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

#         out = torch.cat([out2, out3, out4], dim=1)
#         out = F.relu(self.conv(out))
#         out = self.bn1(self.pooling(out))
#         out = self.bn2(self.linear(out))
#         return out
    
    
# class StatsPool(nn.Module):
    
#     def __init__(self):
#         super(StatsPool, self).__init__()

#     def forward(self, x):
#         x = x.view(x.shape[0], x.shape[1], -1)
#         out = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
#         return out
    
# class X_Vector(nn.Module):
#     def __init__(self, in_dim=80, hidden_dim=512, embedding_size=192):
#         super().__init__()
#         self.layer1 = Conv1dReluBn(in_dim, hidden_dim, kernel_size=5, dilation=1)
#         self.layer2 = Conv1dReluBn(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
#         self.layer3 = Conv1dReluBn(hidden_dim, hidden_dim, kernel_size=3, dilation=3)
#         self.layer4 = Conv1dReluBn(hidden_dim, hidden_dim, kernel_size=1, dilation=1)
#         self.layer5 = Conv1dReluBn(hidden_dim, 1500, kernel_size=1, dilation=1)
        
#         self.pooling = StatsPool()
#         self.fc1 = Conv1dReluBn(3000, embedding_size, kernel_size=1, dilation=1)
#         self.fc2 = Conv1dReluBn(embedding_size, embedding_size, kernel_size=1, dilation=1)

#     def forward(self, x, bottleneck_last=True):
# #         x = x.transpose(1, 2)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.pooling(x)
#         embd = self.fc1(x)
#         x = self.fc2(embd)
#         if bottleneck_last:  
#             return x
#         else:
#             return embd
        
        
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class SE_Res2Block(nn.Module):
    
    def __init__(self,k=3,d=2,s=8,channel=512,bottleneck=128):
        super(SE_Res2Block,self).__init__()
        self.k = k
        self.d = d
        self.s = s
        if self.s == 1:
            self.nums = 1
        else:
            self.nums = self.s - 1
            
        self.channel = channel
        self.bottleneck = bottleneck
        
        self.conv1 = nn.Conv1d(self.channel,self.channel,kernel_size=1,dilation=1)
        self.bn1 = nn.BatchNorm1d(self.channel)
        
        self.convs = []
        self.bns = []
        for i in range(self.s):
            self.convs.append(nn.Conv1d(int(self.channel/self.s), int(self.channel/self.s), kernel_size=self.k, dilation=self.d, padding=self.d, bias=False,padding_mode='reflect'))
            self.bns.append(nn.BatchNorm1d(int(self.channel/self.s)))
            
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        
        self.conv3 = nn.Conv1d(self.channel,self.channel,kernel_size=1,dilation=1)
        self.bn3 = nn.BatchNorm1d(self.channel)
        
        self.fc1 = nn.Linear(self.channel,self.bottleneck,bias=True)
        self.fc2 = nn.Linear(self.bottleneck,self.channel,bias=True)
        
    def forward(self,x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))

        spx = torch.split(out, int(self.channel/self.s), 1)
        for i in range(1,self.nums+1):
            if i==1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = F.relu(self.bns[i](sp))
            if i==1:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.s != 1 :
            out = torch.cat((out, spx[0]),1)
        
        out = F.relu(self.bn3(self.conv3(out)))
        out_mean = torch.mean(out,dim=2)
        s_v = torch.sigmoid(self.fc2(F.relu(self.fc1(out_mean))))
        out = out * s_v.unsqueeze(-1)
        out += residual
        #out = F.relu(out)
        return out


class Classic_Attention(nn.Module):
    def __init__(self,input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
    
    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = torch.tanh(lin_out.bmm(v_view).squeeze(-1))
        attention_weights_normalized = F.softmax(attention_weights,1)
        #attention_weights_normalized = F.softmax(attention_weights)
        return attention_weights_normalized

class Attentive_Statictics_Pooling(nn.Module):
    
    def __init__(self,channel=1536,R_dim_self_att=128):
        super(Attentive_Statictics_Pooling,self).__init__()
        
        self.attention = Classic_Attention(channel,R_dim_self_att)
    
    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance    
    
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling
    
    def forward(self,x):
        attn_weights = self.attention(x)
        stat_pool_out = self.stat_attn_pool(x,attn_weights)
        
        return stat_pool_out
    
class ECAPA_TDNN(nn.Module):
    
    def __init__(self,in_dim=80,hidden_dim=512,scale=8,bottleneck=128,embedding_size=192):
        
        super(ECAPA_TDNN,self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.bottleneck = bottleneck
        self.embedding_size = embedding_size
        
        self.conv1 = nn.Conv1d(in_dim,hidden_dim,kernel_size=5,dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.block1 = SE_Res2Block(k=3,d=2,s=self.scale,channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.block2 = SE_Res2Block(k=3,d=3,s=self.scale,channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.block3 = SE_Res2Block(k=3,d=4,s=self.scale,channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.conv2 = nn.Conv1d(self.hidden_dim*3,self.hidden_dim*3,kernel_size=1,dilation=1)
        
        self.ASP = Attentive_Statictics_Pooling(channel=self.hidden_dim*3,R_dim_self_att=self.bottleneck)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim*3*2)
        
        self.fc = nn.Linear(self.hidden_dim*3*2,self.embedding_size)
        self.bn3 = nn.BatchNorm1d(self.embedding_size)
        
    def forward(self,x):
#         x = x.transpose(1,2)
        y = F.relu(self.bn1(self.conv1(x)))
        y_1 = self.block1(y)
        y_2 = self.block2(y_1)
        y_3 = self.block3(y_2)
        out = torch.cat((y_1, y_2,y_3), 1)
        out = F.relu(self.conv2(out))
        out = self.bn2(self.ASP(out.transpose(1,2)))
        out = self.bn3(self.fc(out))
        return out