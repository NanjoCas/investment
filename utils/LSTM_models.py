#定义了基于LSTM的Actor和Q值模型

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal




# 设置随机种子
torch.manual_seed(666)
# 设置设备
device = torch.device("cpu")     

def Mish(x):
    """
    Mish激活函数
    """
    x = x * (torch.tanh(F.softplus(x)))
    return x
#-----------------------第一次指导-------------------------------
#    可更改：forward,Actor,编码器不一定要用LSTM，TCN，transformer...
#    隐藏层维度，lstm隐藏层维度，lstm层数
#-----------------------------------------------------------------



#----------------------原LSTM_SAC_Actor class,可以运行----------------------------------
# class LSTM_SAC_Actor(nn.Module):
#     def __init__(self, state_dim, time_steps=10, action_dim=1, hidden_list=[128,64]):
#         super(LSTM_SAC_Actor, self).__init__()
#         self.state_dim = state_dim  # 输入状态的维度
#         self.action_dim = action_dim  # 输出动作的维度
#         self.layers = nn.ModuleList()  # 存储神经网络层的列表
#         self.hidden_list = hidden_list  # 隐藏层的维度列表
#         self.lstm_hidden_size = 64  # LSTM隐藏层的维度
#         self.lstm_layer = 2  # LSTM层数

        
#         self.lstm = nn.LSTM(input_size=self.state_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layer, batch_first=True, bidirectional=True)  
#         # 定义LSTM层，输入维度为状态维度，隐藏层维度为LSTM隐藏层维度，层数为LSTM层数，batch_first=True表示输入数据的第一维度为batch大小，bidirectional=True表示双向LSTM
#         insize = self.lstm_hidden_size * 2 + 2  # 输入全连接层的维度，为LSTM隐藏层维度的两倍加上2
#         for outsize in self.hidden_list:
#             fc = nn.Linear(insize, outsize)  # 定义全连接层，输入维度为insize，输出维度为outsize
#             insize = outsize  # 更新下一层全连接层的输入维度
#             self.layers.append(fc)  # 将全连接层添加到神经网络层列表中
#         self.mean_layer = nn.Linear(insize, self.action_dim)  # 定义输出均值的全连接层，输入维度为insize，输出维度为动作维度
#         self.log_std_layer = nn.Linear(insize, self.action_dim)  # 定义输出标准差的全连接层，输入维度为insize，输出维度为动作维度

#     def forward(self, state, moneyAndRatio):     
#         money_ratio = moneyAndRatio[:, -1]  # 提取输入中的资金比例
#         stock_ratio = 1 - money_ratio  # 计算股票比例
#         h0 = torch.zeros(self.lstm_layer * 2, state.size(0), self.lstm_hidden_size).to(device)  # 初始化LSTM隐藏层的初始隐藏状态
#         c0 = torch.zeros(self.lstm_layer * 2, state.size(0), self.lstm_hidden_size).to(device)  # 初始化LSTM隐藏层的初始细胞状态
#         lstm_out, (hn, cn) = self.lstm(state, (h0, c0))  # 运行LSTM层，得到输出和最终的隐藏状态和细胞状态
#         out = lstm_out[:, -1, :].view(lstm_out.size()[0], -1)  # 提取LSTM层的最后一个时间步的输出
#         out = torch.cat((out, moneyAndRatio), -1)  # 将资金比例和LSTM层输出连接起来
#         for layer in self.layers:
#             out = F.leaky_relu(layer(out), 0.2, True)  # 通过激活函数LeakyReLU对每一层进行非线性变换
#         mean = self.mean_layer(out)  # 计算动作的均值
#         log_std = self.log_std_layer(out)  # 计算动作的标准差
#         log_std = torch.clamp(log_std, -20, 2)  # 将标准差限制在[-20, 2]范围内
#         std = log_std.exp()  # 计算标准差的指数形式
#         normal_dist = Normal(mean, std)  # 创建正态分布对象
#         z = normal_dist.sample()  # 从正态分布中采样得到动作
#         true_action = torch.tanh(z)  # 将动作映射到[-1, 1]范围内
#         zeros_action = torch.zeros_like(true_action).to(device)  # 创建与true_action相同形状的全零张量
#         invest_action = torch.max(true_action, zeros_action) * money_ratio + torch.min(true_action, zeros_action) * stock_ratio  # 根据资金比例将动作拆分为投资和持有两部分
#         return true_action, mean, log_std, invest_action
        


#-----------------------更改transformer后的class类--------------------------------------
class TransformerSACActor(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_list=[128, 64], num_heads=8):
        super(TransformerSACActor, self).__init__()
        self.state_dim = state_dim  # 状态维度
        self.action_dim = action_dim  # 动作维度，默认为1
        self.layers = nn.ModuleList()  # 神经网络层列表
        self.hidden_list = hidden_list  # 隐藏层列表
        self.transformer_dim = 64  # Transformer维度
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=state_dim, nhead=num_heads)  # Transformer编码器层
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)  # Transformer编码器
        insize = self.transformer_dim + 2  # 输入尺寸
        for outsize in self.hidden_list:  # 遍历隐藏层列表
            fc = nn.Linear(insize, outsize)  # 全连接层
            insize = outsize
            self.layers.append(fc)  # 添加全连接层到层列表
        self.mean_layer = nn.Linear(insize, self.action_dim)  # 均值层
        self.log_std_layer = nn.Linear(insize, self.action_dim)  # 对数标准差层

    def forward(self, state, moneyAndRatio):
            money_ratio = moneyAndRatio[:, -1]  # 提取资金比例
            stock_ratio = 1 - money_ratio  # 计算股票比例
            transformer_out = self.transformer(state)  # 使用Transformer处理状态
            out = transformer_out.mean(dim=1)  # 求平均值
            out = torch.cat((out, moneyAndRatio), -1)  # 拼接输出和资金比例
            for layer in self.layers:  # 遍历全连接层
                out = F.leaky_relu(layer(out), 0.2, True)  # 使用LeakyReLU激活函数
            mean = self.mean_layer(out)  # 计算均值
            log_std = self.log_std_layer(out)  # 计算对数标准差
            log_std = torch.clamp(log_std, -20, 2)  # 对对数标准差进行截断
            std = log_std.exp()  # 计算标准差
            normal_dist = Normal(mean, std)  # 创建正态分布
            z = normal_dist.sample()  # 采样得到z
            true_action = torch.tanh(z)  # 使用双曲正切函数获得真实动作
            zeros_action = torch.zeros_like(true_action).to(device)  # 创建与true_action相同形状的全零张量
            invest_action = torch.max(true_action, zeros_action) * money_ratio + torch.min(true_action, zeros_action) * stock_ratio  # 计算投资动作
            return true_action, mean, log_std, invest_action  # 返回真实动作、均值、对数标准差和投资动作


#------------------------------原LSTM_Q_Net class,可以运行------------------------------------
# class LSTM_Q_net(nn.Module):
#     def __init__(self, state_dim, time_steps=10, action_dim=1, hidden_list=[128,64]):
#         super(LSTM_Q_net, self).__init__()
#         self.state_dim = state_dim  # 输入状态的维度
#         self.action_dim = action_dim  # 动作的维度
#         self.layers = nn.ModuleList()  # 存储神经网络层的列表
#         self.hidden_list = hidden_list  # 隐藏层的维度列表
#         self.lstm_hidden_size = 64  # LSTM隐藏层的维度
#         self.lstm_layer = 2  # LSTM层数
#         self.lstm = nn.LSTM(input_size=self.state_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layer, batch_first=True, bidirectional=True)  
#         # 定义LSTM层，输入维度为状态维度，隐藏层维度为LSTM隐藏层维度，层数为LSTM层数，batch_first=True表示输入数据的第一维度为batch大小，bidirectional=True表示双向LSTM
#         insize = self.lstm_hidden_size * 2 + 3  # 输入全连接层的维度，为LSTM隐藏层维度的两倍加上3
#         for outsize in self.hidden_list:
#             fc = nn.Linear(insize, outsize)  # 定义全连接层，输入维度为insize，输出维度为outsize
#             insize = outsize  # 更新下一层全连接层的输入维度
#             self.layers.append(fc)  # 将全连接层添加到神经网络层列表中
#         self.q_out_layer = nn.Linear(insize, 1)  # 定义输出Q值的全连接层，输入维度为insize，输出维度为1




#     def forward(self, state, moneyAndRatio, actions):       
#         h0 = torch.zeros(self.lstm_layer * 2, state.size(0), self.lstm_hidden_size).to(device)  # 初始化LSTM隐藏层的初始隐藏状态
#         c0 = torch.zeros(self.lstm_layer * 2, state.size(0), self.lstm_hidden_size).to(device)  # 初始化LSTM隐藏层的初始细胞状态
#         lstm_out, (hn, cn) = self.lstm(state, (h0, c0))  # 运行LSTM层，得到输出和最终的隐藏状态和细胞状态
#         out = lstm_out[:, -1, :].view(lstm_out.size()[0], -1)  # 提取LSTM层的最后一个时间步的输出
#         out = torch.cat((out, moneyAndRatio, actions), -1)  # 将资金比例、动作和LSTM层输出连接起来
#         for layer in self.layers:
#             out = F.leaky_relu(layer(out), 0.2, True)  # 通过激活函数LeakyReLU对每一层进行非线性变换
#         q_value = self.q_out_layer(out)  # 计算Q值
#         return q_value
    

#-------------------------更改TransformerQNet的class------------------------------------
class TransformerQNet(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_list=[128, 64], num_heads=4, num_layers=3):
        super(TransformerQNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_list = hidden_list
        self.num_heads = num_heads
        self.num_layers = num_layers
        # 定义Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=state_dim, nhead=num_heads,batch_first=True),
            num_layers=num_layers
        )
        insize = state_dim + 3  # 输入全连接层的维度，为Transformer输出维度加上3
        self.layers = nn.ModuleList()
        # 构建隐藏层
        for outsize in hidden_list:
            fc = nn.Linear(insize, outsize)
            insize = outsize
            self.layers.append(fc)
        self.q_out_layer = nn.Linear(insize, 1)

    def forward(self, state, moneyAndRatio, actions):
        # Transformer编码器
        transformer_out = self.transformer_encoder(state)
        transformer_out = transformer_out.mean(dim=1)  # 取平均作为输出，根据具体任务可能需要调整
        # 与moneyAndRatio和actions拼接
        out = torch.cat((transformer_out, moneyAndRatio, actions), -1)
        # 全连接层
        for layer in self.layers:
            out = F.leaky_relu(layer(out), 0.2, True)
        # 输出层
        q_value = self.q_out_layer(out)
        return q_value
    


#--------以下为gpt生成的示例用法(没有用),运行时[无需]取消注释-----------
# # 示例用法
# state_dim = 64  # 你的状态维度
# model = TransformerQNet(state_dim)

# # 创建一个输入样本
# state = torch.rand((batch_size, sequence_length, state_dim))
# money_and_ratio = torch.rand((batch_size, 3))  # 假设资金比例有3个维度
# actions = torch.rand((batch_size, action_dim))

# # 前向传播
# q_values = model(state, money_and_ratio, actions)
# print(q_values)
