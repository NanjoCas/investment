#定义量化交易策略和回测过程

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import numpy as np
import pandas as pd
# import tushare as ts
import backtrader as bt
from ASAC_Trader import ASAC_Trader_wrapper
import matplotlib.pyplot as plt
import copy

###############################  training  ####################################


KEEP_DAY = 6
TIME_STEPS = 55
STATE_DIM = 16

trader = ASAC_Trader_wrapper(state_dim=STATE_DIM, action_dim=1, time_steps=TIME_STEPS,eval_mode=True)

# 创建策略
class TestStrategy(bt.Strategy):
    params = dict(maperiod=55)

    def log(self, txt, dt=None):
        '''用于记录日志的函数'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
 
    def __init__(self):
        # 计算成交量的指数移动平均线
        volume_ema = bt.indicators.EMA(self.data.volume, period=self.p.maperiod, plot=False)
        # 计算收盘价的指数移动平均线
        self.close_ema = bt.indicators.EMA(self.data.close, period=self.p.maperiod, plot=False)
        # 计算成交量比率
        self.volume_ratio = self.data.volume / (volume_ema)
        # 计算收盘价比率
        self.close_ratio = self.data.close / (self.close_ema) 
        # 计算开盘价比率
        self.open_ratio = self.data.open / (self.close_ema) 
        # 计算最高价比率
        self.high_ratio = self.data.high / (self.close_ema) 
        # 计算最低价比率
        self.low_ratio = self.data.low / (self.close_ema) 
        # 状态列表
        self.state_list = []
        # 上一个状态的价值
        self.last_state_value = 0
        # 上一个状态
        self.last_state = np.array([[]])
        # 时间步长
        self.time_steps = TIME_STEPS
        # 订单
        self.order = None
        # 数据存储
        self.data_store = []
        # 当前冷却天数
        self.current_cold_day = 0
        # 上一个持有股票的数量
        self.last_stock_size = 0
        # 上一个现金
        self.last_money = 0
        # 当前持有股票的数量
        self.current_stock_size = 0
        # 平均股票成本
        self.avg_stock_cost = self.data.close[0]

    def notify_order(self, order):
        # 如果订单状态为已提交或已接受，则返回
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 如果订单状态为已完成
        if order.status in [order.Completed]:
            # 如果是买入订单
            if order.isbuy(): 
                # 记录买入价格
                self.buyprice = order.executed.price
                # 记录买入手续费
                self.buycomm = order.executed.comm
            else:  
                pass
        # 如果订单状态为已取消、保证金或拒绝
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # 输出订单状态日志
            self.log(order.status)
        # 清空订单
        self.order = None

        
#    def notify_cashvalue(self, cash, value):
#        self.log('Cash %s Value %s' % (cash, value)) 
    
#    def notify_trade(self, trade):
#        if not trade.isclosed:
#            return
#        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
#                 (trade.pnl, trade.pnlcomm))
 
    def store_data(self, current_state, current_action, current_moneyRatio, chance_return):
        # 创建存储数据的字典
        sg_data = dict(
            current_state= current_state,  # 当前状态
            current_moneyRatio = current_moneyRatio,  # 当前资金比率
            current_action = current_action,  # 当前动作
            chance_return = chance_return  # 预期收益率
        )
        # 将数据存储到数据存储列表中
        self.data_store.append(sg_data)        
        # 如果数据存储列表中有两个数据
        if len(self.data_store) == 2:
            # 更新总步数
            trader.total_step = trader.total_step + 1  # 交易员增加经验
            # 获取上一个经验和下一个经验
            last_experience = self.data_store[0]
            next_experience = self.data_store[1]
            # 计算奖励
            sg_return = 10 * next_experience['chance_return'] + 0.03 * np.abs(last_experience['current_action'])  # ASAC的单一奖励
            # 添加经验
            trader.add_experience(
                state=last_experience['current_state'],  # 上一个状态
                moneyRatio=last_experience['current_moneyRatio'],  # 上一个资金比率
                action=last_experience['current_action'],  # 上一个动作
                reward=sg_return,  # 奖励
                next_state=next_experience['current_state'],  # 下一个状态
                next_moneyRatio=next_experience['current_moneyRatio'],  # 下一个资金比率
                done=[False]  # 完成标志
            )                    
            # 删除生成过渡元组后的第一行数据
            self.data_store = self.data_store[1:]


    def collect_status(self):            
    # 获取上一个状态的价值
        self.last_state_value = self.broker.getvalue()
        # 获取上一个现金
        self.last_money = self.broker.getcash()
        # 获取上一个持有股票的数量
        self.last_stock_size = self.position.size
        # 如果持有股票的数量大于0
        if self.position.size > 0:
            # 计算平均股票成本
            self.avg_stock_cost = (self.broker.getvalue() - self.broker.getcash() ) / self.position.size

             
    def next(self):
        # 当前冷却天数减1
        self.current_cold_day = self.current_cold_day - 1
        # 如果有订单存在，则返回
        if self.order:
            return 
        # 计算平均成本比率
        average_cost_ratio = self.avg_stock_cost / self.close_ema[0]
        # 获取当前现金和价值
        current_money = self.broker.getcash()
        current_value = self.broker.getvalue()
        # 计算资金比率
        money_ratio = current_money * 1.0 / (current_value)
        # 构建当前资金比率数组
        current_moneyRatio = np.array([average_cost_ratio, money_ratio])
        # 构建当前状态数组
        current_status = [self.close_ratio[0], self.open_ratio[0], self.volume_ratio[0], self.high_ratio[0], self.low_ratio[0]]
        # 将当前状态添加到状态列表中
        self.state_list.append(current_status)
        # 限制状态列表的长度
        self.state_list = self.state_list[-self.time_steps:]
        # 将状态列表转换为numpy数组
        current_state = np.array(self.state_list).reshape(1, -1, STATE_DIM)
        # 选择动作
        source_action, invest_action = trader.choose_action(current_state, current_moneyRatio)
        source_action, invest_action = source_action.item(), invest_action.item()
        # 如果上一个状态的长度小于时间步长，则更新上一个状态的值并返回，不执行任何操作
        if self.last_state.shape[1] < self.time_steps:
            self.last_state_value = self.broker.getvalue()
            self.last_state = copy.deepcopy(current_state)
            self.last_money = self.broker.get_cash()
            return
        else:
            # 如果当前冷却天数大于0，则返回，不执行任何操作
            if self.current_cold_day > 0:
                return
            else:
                # 重置当前冷却天数
                self.current_cold_day = KEEP_DAY
                # 计算上一个状态和当前状态的收益率差
                last_return = current_value * 1.0 / (self.last_state_value) - 1 
                if_no_action_value = self.last_stock_size * self.data.close[0] + self.last_money
                no_action_return = if_no_action_value * 1.0 / (self.last_state_value) - 1 
                chance_return = last_return - no_action_return
                # 存储数据
                self.store_data(current_state, source_action, current_moneyRatio, chance_return)
                # 收集状态信息
                self.collect_status() 
                # 交易员进行学习
                trader.trader_learn()
                # 执行动作
                if invest_action > 0:              
                    buy_size = int(invest_action * current_value / (self.data.close[0]) )
                    buy_size = int(buy_size * 0.90/100) * 100 ## 以防开盘价大于收盘价
                    self.order = self.buy(size=buy_size)
                elif invest_action < 0:        
                    sell_size = int(-invest_action * (current_value) / (self.data.close[0]) )
                    sell_size = int(sell_size/100 ) * 100
                    self.order = self.sell(size=sell_size)

            

        



final_value_list = []

# 运行训练函数
def run_train():
    # 数据文件夹路径
    data_folder = 'data'
    # 获取文件名列表并随机打乱顺序
    file_name_list = os.listdir(data_folder)
    np.random.shuffle(file_name_list)
    # 遍历文件名列表
    for file_index in range(len(file_name_list)):
        try:
            # 获取文件名和文件路径
            file_name = file_name_list[file_index]
            file_path = os.path.join(data_folder, file_name)
            # 读取数据文件
            dataframe = pd.read_csv(file_path) 
            # 设置索引为日期
            dataframe.index = dataframe['date'].apply(lambda x: pd.Timestamp(x))
            # 将0值替换为NaN
            dataframe = dataframe.where(dataframe != 0)
            # 使用前向填充方法填充NaN值
            dataframe = dataframe.fillna(method='ffill')
            # 添加空的持仓量列
            dataframe['openinterest'] = 0 
            # 如果数据行数小于1200，则跳过该文件
            if dataframe.shape[0] < 1200:
                continue 
            # 创建Cerebro实例
            cerebro = bt.Cerebro()
            # 添加策略
            cerebro.addstrategy(TestStrategy)
            # 创建数据源
            data = bt.feeds.PandasData(dataname=dataframe,                               
                                       fromdate=dataframe.index[-300].to_pydatetime(),                               
                                       todate=dataframe.index[-1].to_pydatetime()                              
                                      )
            # 添加数据源
            cerebro.adddata(data)
            # 设置初始现金
            cerebro.broker.setcash(800000)
            # 设置是否允许卖空
            cerebro.broker.set_shortcash(shortcash=False)  # 这很重要
            # 设置佣金
            cerebro.broker.setcommission(commission=0.001)
            # 输出初始投资组合价值
            print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
            # 运行回测
            cerebro.run()
            # 输出最终投资组合价值
            print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
            # 计算收益率
            reward = cerebro.broker.getvalue() * 1.0 / 800000 - 1
            # 添加到最终价值列表中
            final_value_list.append(reward)
        except:
            print(file_name_list[file_index])
            continue
        # 绘制图表
        cerebro.plot()

# 运行训练函数
run_train()

# 绘制最终价值列表
plt.plot(final_value_list)
plt.show()

# 输出平均收益率
print(np.mean(final_value_list))
