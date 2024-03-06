#定义了经验回放缓冲区类ReplayBuffer和LSTM_ReplayBuffer，用于存储和采样经验数据
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



device = torch.device("cpu")
    
class ReplayBuffer(object):
    """
    用于代理的简单FIFO经验回放缓冲区。
    """
    def __init__(self, obs_dim, act_dim, size):
        # 创建观察缓冲区
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        # 创建下一个观察缓冲区
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        # 创建动作缓冲区
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        # 创建奖励缓冲区
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        # 创建完成标志缓冲区
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        # 创建指针、大小和最大大小
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, act, rew, next_obs, done):
        # 添加经验到缓冲区
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        # 更新指针和大小
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=64):
        # 随机采样经验
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=torch.Tensor(self.obs1_buf[idxs]).to(device),
                    obs2=torch.Tensor(self.obs2_buf[idxs]).to(device),
                    acts=torch.Tensor(self.acts_buf[idxs]).to(device),
                    rews=torch.Tensor(self.rews_buf[idxs]).to(device),
                    done=torch.Tensor(self.done_buf[idxs]).to(device))
    

#------------------原LSTM_ReplayBuffer,可直接运行------------------------------      
# class LSTM_ReplayBuffer(object):
#     """
#     用于代理的简单FIFO经验回放缓冲区。
#     """
#     def __init__(self, obs_dim, act_dim, size, time_steps=10):
#         # 创建观察缓冲区
#         self.obs1_buf = np.zeros([size, time_steps, obs_dim], dtype=np.float32)
#         # 创建资金和比率缓冲区
#         self.moneyAndRatio1 = np.zeros([size, 2], dtype=np.float32) # 状态向量，资金及其比率
#         # 创建下一个观察缓冲区
#         self.obs2_buf = np.zeros([size, time_steps, obs_dim], dtype=np.float32)
#         # 创建下一个资金和比率缓冲区
#         self.next_moneyAndRatio = np.zeros([size, 2], dtype=np.float32)
#         # 创建动作缓冲区
#         self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
#         # 创建奖励缓冲区
#         self.rews_buf = np.zeros([size, 1], dtype=np.float32)
#         # 创建完成标志缓冲区
#         self.done_buf = np.zeros([size, 1], dtype=np.float32)
#         # 创建指针、大小和最大大小
#         self.ptr, self.size, self.max_size = 0, 0, size

#     def add(self, obs, moneyAndRatio1, act, rew, next_obs, moneyAndRatio2, done):
#         # 添加经验到缓冲区
#         self.obs1_buf[self.ptr] = obs
#         self.moneyAndRatio1[self.ptr] = moneyAndRatio1
#         self.obs2_buf[self.ptr] = next_obs
#         self.next_moneyAndRatio[self.ptr] = moneyAndRatio2
#         self.acts_buf[self.ptr] = act
#         self.rews_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         # 更新指针和大小
#         self.ptr = (self.ptr+1) % self.max_size
#         self.size = min(self.size+1, self.max_size)

#     def sample(self, batch_size=64):
#         # 随机采样经验
#         idxs = np.random.randint(0, self.size, size=batch_size)
#         return dict(obs1=torch.Tensor(self.obs1_buf[idxs]).to(device),
#                     moneyRatio=torch.Tensor(self.moneyAndRatio1[idxs]).to(device),
#                     obs2=torch.Tensor(self.obs2_buf[idxs]).to(device),
#                     next_moneyRatio=torch.Tensor(self.next_moneyAndRatio[idxs]).to(device),                   
#                     acts=torch.Tensor(self.acts_buf[idxs]).to(device),
#                     rews=torch.Tensor(self.rews_buf[idxs]).to(device),
#                     done=torch.Tensor(self.done_buf[idxs]).to(device))
    



#---------------------修改后的Transformer ReplayBuffer class----------------------------
class TransformerReplayBuffer(object):
    """
    用于代理的简单FIFO经验回放缓冲区。
    """
    def __init__(self, obs_dim, act_dim, size):
        # 创建观察缓冲区
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        # 创建资金和比率缓冲区
        self.moneyAndRatio1 = np.zeros([size, 2], dtype=np.float32) # 状态向量，资金及其比率
        # 创建下一个观察缓冲区
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        # 创建下一个资金和比率缓冲区
        self.next_moneyAndRatio = np.zeros([size, 2], dtype=np.float32)
        # 创建动作缓冲区
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        # 创建奖励缓冲区
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        # 创建完成标志缓冲区
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        # 创建指针、大小和最大大小
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, moneyAndRatio1, act, rew, next_obs, moneyAndRatio2, done):
        # 添加经验到缓冲区
        self.obs1_buf[self.ptr] = obs
        self.moneyAndRatio1[self.ptr] = moneyAndRatio1
        self.obs2_buf[self.ptr] = next_obs
        self.next_moneyAndRatio[self.ptr] = moneyAndRatio2
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        # 更新指针和大小
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=64):
        # 随机采样经验
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=torch.Tensor(self.obs1_buf[idxs]).to(device),
                    moneyRatio=torch.Tensor(self.moneyAndRatio1[idxs]).to(device),
                    obs2=torch.Tensor(self.obs2_buf[idxs]).to(device),
                    next_moneyRatio=torch.Tensor(self.next_moneyAndRatio[idxs]).to(device),                   
                    acts=torch.Tensor(self.acts_buf[idxs]).to(device),
                    rews=torch.Tensor(self.rews_buf[idxs]).to(device),
                    done=torch.Tensor(self.done_buf[idxs]).to(device))




