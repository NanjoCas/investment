#定义了交易员类，包括选择动作、评估、学习、保存和加载模型等方法
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# from utils.util import LSTM_ReplayBuffer
from utils.util import TransformerReplayBuffer

# from utils.LSTM_models import LSTM_SAC_Actor, LSTM_Q_net
from utils.LSTM_models import TransformerSACActor, TransformerQNet

from torch.distributions import Normal




torch.manual_seed(666)
device = torch.device("cpu")     

class ASAC_Trader_wrapper(object):
    def __init__(self, state_dim, action_dim, time_steps=10, eval_mode=False, gamma=0.99, tau=0.05, batch_size=64, buffer_size=40000, start_learn_step=10000):
        """
        初始化ASAC_Trader_wrapper对象。

        参数：
        state_dim：int，状态维度。
        action_dim：int，动作维度。
        time_steps：int，时间步数，默认为10。
        eval_mode：bool，是否处于评估模式，默认为False。
        gamma：float，折扣因子，默认为0.99。
        tau：float，软更新因子，默认为0.05。
        batch_size：int，批次大小，默认为64。
        buffer_size：int，回放缓冲区大小，默认为40000。
        start_learn_step：int，开始学习的步数，默认为10000。
        """
        # 设置模型路径
        self.trader_model_path = './weights/LSTM_ASAC/trader_model.pth'
        self.q_net1_model_path = './weights/LSTM_ASAC/q_net1_model.pth'
        self.q_net2_model_path = './weights/LSTM_ASAC/q_net2_model.pth'

        # 设置参数
        self.action_dim, self.state_dim, self.time_steps = action_dim, state_dim, time_steps
        self.batch_size, self.buffer_size, self.gamma, self.tau = batch_size, buffer_size, gamma, tau

        # 创建回放缓冲区
        self.replay_buffer = TransformerReplayBuffer(self.state_dim, self.action_dim, self.buffer_size, time_steps=time_steps)

        # 初始化步数和开始学习步数
        self.total_step, self.start_learn_step = 0, start_learn_step

        # 设置评估模式
        self.eval_mode = eval_mode

        # 创建交易员和Q网络
        # self.trader = LSTM_SAC_Actor(self.state_dim, self.action_dim).to(device)
        # self.Q_net1 = LSTM_Q_net(self.state_dim, self.action_dim).to(device)
        # self.Q_net2 = LSTM_Q_net(self.state_dim, self.action_dim).to(device)
        # self.Q_net_target1 = LSTM_Q_net(self.state_dim, self.action_dim).to(device)
        # self.Q_net_target2 = LSTM_Q_net(self.state_dim, self.action_dim).to(device)

        self.trader = TransformerSACActor(self.state_dim, self.action_dim).to(device)
        self.Q_net1 = TransformerQNet(self.state_dim, self.action_dim).to(device)
        self.Q_net2 = TransformerQNet(self.state_dim, self.action_dim).to(device)
        self.Q_net_target1 = TransformerQNet(self.state_dim, self.action_dim).to(device)
        self.Q_net_target2 = TransformerQNet(self.state_dim, self.action_dim).to(device)

        # 创建损失函数
        self.Q_net1_criterion = nn.MSELoss()
        self.Q_net2_criterion = nn.MSELoss()

        # 创建Q网络参数列表
        self.q_net_parameter = list(self.Q_net1.parameters()) + list(self.Q_net2.parameters())

        # 创建优化器
        self.trader_optimizer = optim.Adam(self.trader.parameters(), lr=0.0003)
        self.Qnet_optimizer = optim.Adam(self.q_net_parameter, lr=0.0003)

        # 设置目标熵和alpha值
        self.target_entropy = - np.prod((self.action_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device).to(device)
        self.alpha = 0.2
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003)

        # 加载模型
        self.load_model() 
        
        
        
            
    def choose_action(self, state, moneyAndRatio):
        # 将state和moneyAndRatio转换为torch.FloatTensor类型，并移动到设备上
        state = torch.FloatTensor(state).to(device)
        moneyAndRatio = torch.FloatTensor(moneyAndRatio).unsqueeze(0).to(device)
        # 调用self.trader函数获取true_action和invest_action
        true_action, _,  _, invest_action = self.trader(state, moneyAndRatio)
        return true_action.cpu().detach().numpy(), invest_action.cpu().detach().numpy() 

    def evaluate(self, state, moneyAndRatio ):
        # 调用self.trader函数获取mean和log_std
        _, mean,  log_std, _ = self.trader(state, moneyAndRatio)
        std = log_std.exp()
        normal_dist = Normal(mean, std)
        dist = Normal(0,1)
        z = dist.sample().to(device)
        action_tanh = torch.tanh( mean + std * z )
        log_prob = normal_dist.log_prob(mean + std * z) - torch.log( 1 -  action_tanh**2  + 1e-8 )
        return action_tanh, log_prob

    def add_experience(self, state, moneyRatio, action, reward, next_state, next_moneyRatio, done):
        # 将经验添加到replay_buffer中
        self.replay_buffer.add(state, moneyRatio, action, reward, next_state, next_moneyRatio, done)

    def soft_update(self):
        # 使用soft update算法更新Q_net1和Q_net_target1的参数
        for params, target_params in zip(self.Q_net1.parameters(),self.Q_net_target1.parameters()):
            target_params.data.copy_(params * self.tau + target_params * (1 - self.tau))
        # 使用soft update算法更新Q_net2和Q_net_target2的参数
        for params, target_params in zip(self.Q_net2.parameters(),self.Q_net_target2.parameters()):
            target_params.data.copy_(params * self.tau + target_params * (1 - self.tau))

    def trader_learn(self):
        if self.total_step > self.start_learn_step and not self.eval_mode:
            # 从replay_buffer中采样一个batch的数据
            batch = self.replay_buffer.sample(self.batch_size) 
            state1, state2, actions, rewards, _ = batch['obs1'],batch['obs2'],batch['acts'],batch['rews'],batch['done']
            moneyAndRatio1, next_moneyAndRatio = batch['moneyRatio'], batch['next_moneyRatio']
            # 使用Q_net1和Q_net2计算q1和q2
            q1 = self.Q_net1(state1, moneyAndRatio1, actions)
            q2 = self.Q_net2(state1, moneyAndRatio1, actions)
            # 使用evaluate函数计算next_pi和next_pi_log_prob
            next_pi, next_pi_log_prob = self.evaluate(state2, next_moneyAndRatio)
            # 使用Q_net_target1和Q_net_target2计算next_pi_q1和next_pi_q2
            next_pi_q1 = self.Q_net_target1(state2, next_moneyAndRatio, next_pi)
            next_pi_q2 = self.Q_net_target2(state2, next_moneyAndRatio, next_pi)
            # 取next_pi_q1和next_pi_q2的最小值
            next_pi_q = torch.min(next_pi_q1, next_pi_q2)
            # 计算next_v_value
            next_v_value = next_pi_q - self.alpha * next_pi_log_prob
            # 计算q_target
            q_target = rewards + self.gamma * next_v_value
            # 计算q1_loss和q2_loss
            q1_loss = self.Q_net1_criterion(q1, q_target.detach())
            q2_loss = self.Q_net2_criterion(q2, q_target.detach())
            q_net_loss = q1_loss + q2_loss
            # 使用evaluate函数计算pi和pi_log_prob
            pi,  pi_log_prob  = self.evaluate(state1, moneyAndRatio1)
            # 使用Q_net_target1和Q_net_target2计算pi_q1和pi_q2
            pi_q1 = self.Q_net_target1(state1, moneyAndRatio1, pi)
            pi_q2 = self.Q_net_target2(state1, moneyAndRatio1, pi)
            # 取pi_q1和pi_q2的最小值
            pi_q = torch.min(pi_q1, pi_q2)
            # 计算trader_loss
            trader_loss = (self.alpha * pi_log_prob - pi_q).mean()
            # 计算alpha_loss
            alpha_loss = - (self.log_alpha * (pi_log_prob + self.target_entropy).detach()).mean()
            # 清除Qnet_optimizer的梯度
            self.Qnet_optimizer.zero_grad()
            # 反向传播计算梯度
            q_net_loss.backward()
            # 更新Q_net1和Q_net2的参数
            self.Qnet_optimizer.step()
            # 清除trader_optimizer的梯度
            self.trader_optimizer.zero_grad()
            # 反向传播计算梯度
            trader_loss.backward()
            # 更新trader的参数
            self.trader_optimizer.step()
            # 清除alpha_optimizer的梯度
            self.alpha_optimizer.zero_grad()
            # 反向传播计算梯度
            alpha_loss.backward()
            # 更新alpha的参数
            self.alpha_optimizer.step()
            # 更新alpha的值
            self.alpha = self.log_alpha.exp()
            # 执行soft_update
            self.soft_update()
            # 每10000步保存模型
            if (self.total_step + 1) % 10000 == 0:
                self.save_model()

    def save_model(self):
        # 保存模型
        torch.save(self.trader,self.trader_model_path)
        torch.save(self.Q_net_target1,self.q_net1_model_path)
        torch.save(self.Q_net_target2,self.q_net2_model_path)

    def load_model(self):
        try:
            # 加载模型
            self.trader.load_state_dict(torch.load(self.trader_model_path,map_location=device).state_dict())
            self.Q_net1.load_state_dict(torch.load(self.q_net1_model_path,map_location=device).state_dict())
            self.Q_net2.load_state_dict(torch.load(self.q_net2_model_path,map_location=device).state_dict())
            self.Q_net_target1.load_state_dict(torch.load(self.q_net1_model_path,map_location=device).state_dict())
            self.Q_net_target2.load_state_dict(torch.load(self.q_net2_model_path,map_location=device).state_dict())
            print('load model success')
        except:
            print('load model error')