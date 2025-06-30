import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

FILTERS = 16
KERNEL_SIZE = 1
BLOCKS = 3

# 残差块
class BasicBlock(nn.Module):
	def __init__(self):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(FILTERS, FILTERS, kernel_size=3,
							stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(FILTERS)
		self.conv2 = nn.Conv2d(FILTERS, FILTERS, kernel_size=3,
							stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(FILTERS)

	def forward(self, x):
		residual = x
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual
		out = F.relu(out)
		return out

class GoNNet(nn.Module):
    def __init__(self, n, args):
        # game params
        self.board_x = n
        self.board_y = n
        self.action_size = n*n
        self.args = args

        super(GoNNet, self).__init__()
        self.passed = False
        # Feature Extractor
        # 特征提取起始点，把输入转为 FILTERS=16 个通道
        self.conv1 = nn.Conv2d(self.args.board_feature_channel + 1, FILTERS, stride=1,
							kernel_size=KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(FILTERS)
        # BLOCKS=3 个残差块
        for block in range(BLOCKS):
            setattr(self, "res{}".format(block), \
				BasicBlock())
        # Policy Head策略头
        if(self.args.policy != "AOA"):
            # Conv2d(16 → 2) 提取策略特征
            self.convPolicy = nn.Conv2d(FILTERS, 2, kernel_size=1)
            self.bnPolicy = nn.BatchNorm2d(2)
            # 展平后输入到 fc，输出维度是所有可能落子位置数（action_size = n*n）
            self.fc = nn.Linear((self.board_x + 2) * (self.board_y+2) * 2,
            					self.action_size)
            self.softmax = nn.Softmax(dim=1)
            
        # Value Head 价值头
        # 估计当前局面的价值 v
        self.convValue = nn.Conv2d(FILTERS, 1, kernel_size=1)
        self.bnValue = nn.BatchNorm2d(1)
        self.fcValue1 = nn.Linear((self.board_x + 2) * (self.board_y+2) , 64)
        self.fcValue2 = nn.Linear(64, 1)

    def feature(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(BLOCKS - 1):
            x = getattr(self, "res{}".format(block))(x)

        feature_map = getattr(self, "res{}".format(BLOCKS - 1))(x)
        return feature_map

    def forward(self, s):
        
        feature = self.feature(s)
        x = F.relu(self.bnPolicy(self.convPolicy(feature)))
        x = x.view(-1, (self.board_x + 2) * (self.board_y+2) * 2)
        pi = self.fc(x)
        # log_softmax 输出概率的对数，便于训练时使用 cross-entropy loss
        pi = F.log_softmax(pi, dim=1)

        x = F.relu(self.bnValue(self.convValue(feature)))
        x = x.view(-1, (self.board_x + 2) * (self.board_y+2))
        x = F.relu(self.fcValue1(x))
        v = self.fcValue2(x)
        # 输出通过 tanh 再缩放为 0~1 表示胜率
        v_out = (torch.tanh(v) + 1)/2
        return pi, v_out