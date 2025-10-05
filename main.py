"""
# @Author: ywwsnowboy@outlook.com
# @Date: 2025-09-22 23:13:48
# @LastEditors: ywwsnowboy@outlook.com
# @LastEditTime: 2025-09-24 15:37:29
# @FilePath: /DQN_cartpole/main.py
# @Description: 
# @Input: 
# @Output: 
# @
# @Copyright (c) 2025 by ywwsnowboy@outlook.com, All Rights Reserved. 
"""

# 导入PyTorch核心库:张量操作、优化器、神经网络层
import torch
import torch.optim as optim  # 优化器模块(Adam等)
import torch.nn.functional as F_torch  # 激活函数和损失函数
import torch.nn as nn_torch  # 神经网络构建模块
import numpy as np  # 数值计算库
import gymnasium as gym  # 强化学习环境库

# ======== 超参数配置 ========
Batch_size = 32  # 经验回放缓冲区每次采样样本数量
Lr = 0.01  # 学习率(控制参数更新步长)
Epsilon = 0.9  # ε-greedy策略中的探索率(90%概率选择最优动作)
Gamma = 0.9  # 折扣因子(衡量未来奖励的重要性)
Target_replace_iter = 100  # 目标网络更新频率(每100步更新一次)
Memory_capacity = 2000  # 经验回放缓冲区容量

# ======== 创建强化学习环境 ========
env = gym.make('CartPole-v1', render_mode="human")  # 创建CartPole环境(带渲染)
env = env.unwrapped  # 移除包装器(直接访问原始环境属性)
N_actions = env.action_space.n  # 动作空间大小(CartPole有2个动作:左/右)
N_states = env.observation_space.shape[0]  # 状态空间维度(4个状态:位置、速度、角度、角速度)
# 判断动作空间类型(离散动作则ENV_A_SHAPE=0)
# print(isinstance(env.action_space.sample(), int))
# print(env.action_space.sample().shape)
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape # 这行代码未用上

# ======== 定义神经网络模型 ========
class Net(nn_torch.Module):
    """DQN的Q值网络(评估网络) - 用于估计状态-动作价值"""
    def __init__(self):
        super(Net, self).__init__()  # 初始化父类(nn_torch.Module) 这行代码的作用是调用父类的构造函数，确保父类nn_torch.Module被正确初始化
        # 第一层全连接:输入N_states维状态，输出50维
        self.fc1 = nn_torch.Linear(N_states, 50)
        # 初始化权重(均值0，标准差0.1的正态分布)
        self.fc1.weight.data.normal_(0, 0.1)
        # 第二层全连接:输入50维，输出N_actions维(每个动作的Q值)
        self.out = nn_torch.Linear(50, N_actions)
        # 初始化输出层权重
        self.out.weight.data.normal_(0, 0.1) #

    def forward(self, x):
        """前向传播:状态→Q值估计"""
        x = self.fc1(x)  # 通过第一层全连接
        x = F_torch.relu(x)  # 应用ReLU激活函数
        actions_value = self.out(x)  # 通过输出层
        return actions_value  # 返回每个动作的Q值

# ======== DQN核心算法实现 ========
class DQN(object):
    """DQN(Deep Q-Network)算法实现 - 核心训练逻辑"""
    def __init__(self):
        # 创建评估网络(实时更新)和目标网络(定期更新)
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # 学习步数计数器(用于目标网络更新)
        self.memory_counter = 0  # 经验回放缓冲区计数器
        # 初始化经验回放缓冲区(容量×(当前状态+动作+奖励+下一状态))
        self.memory = np.zeros((Memory_capacity, N_states * 2 + 2))
        # Adam优化器(学习率Lr)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=Lr)
        # 均方误差损失函数(用于Q值拟合)
        self.loss_func = nn_torch.MSELoss()

    def choose_action(self, x):
        """选择动作(ε-greedy策略)"""
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # 将状态转换为PyTorch张量(添加batch维度)
        # 以Epsilon概率探索(随机选择动作)
        if np.random.uniform() < Epsilon:
            action_value = self.eval_net(x)  # 评估网络预测Q值
            action = torch.max(action_value, 1)[1].item()  # 选择Q值最大的动作
        else:
            action = np.random.randint(0, N_actions)  # 随机选择动作
        return action

    def store_transition(self, s, a, r, s_):
        """存储经验(状态→动作→奖励→下一状态)"""
        transition = np.hstack((s, [a, r], s_)) # 合并状态、动作、奖励、下一状态为单个过渡向量
        index = self.memory_counter % Memory_capacity # 计算存储位置(循环缓冲区)
        self.memory[index, :] = transition  # 存储到缓冲区
        self.memory_counter += 1  # 更新计数器

    def learn(self):
        """执行一次学习(更新网络参数)"""
        # 每 Target_replace_iter 步更新目标网络
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1  # 更新学习步数

        # 从经验回放缓冲区随机采样 Batch_size 个样本
        sample_index = np.random.choice(Memory_capacity, Batch_size)
        b_memory = self.memory[sample_index, :]

        # 将采样数据转换为PyTorch张量
        b_s = torch.tensor(b_memory[:, :N_states], dtype=torch.float32)  # 当前状态
        b_a = torch.tensor(b_memory[:, N_states:N_states+1].astype(int), dtype=torch.long)  # 动作(整数索引)
        b_r = torch.tensor(b_memory[:, N_states+1:N_states+2], dtype=torch.float32)  # 奖励
        b_s_ = torch.tensor(b_memory[:, -N_states:], dtype=torch.float32)  # 下一状态

        # 评估网络预测当前状态的Q值(仅取动作a对应的Q值)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # 目标网络预测下一状态的最大Q值(不参与梯度计算)
        q_next = self.target_net(b_s_).detach()
        # 计算目标Q值:当前奖励 + Gamma * 下一状态最大Q值
        q_target = b_r + Gamma * q_next.max(1)[0].view(Batch_size, 1)

        # 计算损失(预测Q值与目标Q值的MSE)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        self.optimizer.step()  # 参数更新

# ======== 主训练循环 ========
def main():
    dqn = DQN()  # 初始化DQN算法
    print('\nCollecting experience...')  # 提示开始收集经验

    for i_episode in range(400):  # 训练400轮
        # 重置环境并获取初始状态(返回状态和环境信息)
        s, info = env.reset()
        ep_r = 0  # 当前回合累计奖励

        while True:  # 每轮游戏循环
            env.render()  # 渲染环境(可视化)
            a = dqn.choose_action(s)  # 选择动作
            # 执行动作并获取结果(返回下一状态、奖励、终止标志、截断标志、环境信息)
            s_, r, terminated, truncated, info = env.step(a)
            # 重定义奖励函数(基于当前状态s，而非下一状态s_)
            x, x_dot, theta, theta_dot = s  # 提取当前状态
            # 位置奖励:越靠近中心奖励越高
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # 角度奖励:越接近垂直奖励越高
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2  # 组合奖励
            # 存储经验(当前状态、动作、奖励、下一状态)
            dqn.store_transition(s, a, r, s_)
            ep_r += r  # 累计奖励

            # 经验回放缓冲区满后开始学习
            if dqn.memory_counter > Memory_capacity:
                dqn.learn()
                # 如果回合结束(任务完成或超时)，打印结果
                if terminated or truncated:
                    print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))

            # 检查回合是否结束(任务完成或超时)
            if terminated or truncated:
                break

            s = s_  # 更新当前状态为下一状态

    env.close()  # 关闭环境

# ======== 程序入口 ========
if __name__ == '__main__':
    main()  # 执行主函数