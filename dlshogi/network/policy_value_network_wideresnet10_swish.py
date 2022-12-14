import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


k = 192
fcl = 256 # fully connected layers
class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand
        self.l2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l4 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l5 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l6 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l7 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l8 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l9 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l10 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l11 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l12 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l13 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l14 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l15 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l16 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l17 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l18 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l19 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l20 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l21 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # policy network
        self.l22 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l22_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)
        # value network
        self.l22_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.l23_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l24_v = nn.Linear(fcl, 1)
        self.norm1 = nn.BatchNorm2d(k)
        self.norm2 = nn.BatchNorm2d(k)
        self.norm3 = nn.BatchNorm2d(k)
        self.norm4 = nn.BatchNorm2d(k)
        self.norm5 = nn.BatchNorm2d(k)
        self.norm6 = nn.BatchNorm2d(k)
        self.norm7 = nn.BatchNorm2d(k)
        self.norm8 = nn.BatchNorm2d(k)
        self.norm9 = nn.BatchNorm2d(k)
        self.norm10 = nn.BatchNorm2d(k)
        self.norm11 = nn.BatchNorm2d(k)
        self.norm12 = nn.BatchNorm2d(k)
        self.norm13 = nn.BatchNorm2d(k)
        self.norm14 = nn.BatchNorm2d(k)
        self.norm15 = nn.BatchNorm2d(k)
        self.norm16 = nn.BatchNorm2d(k)
        self.norm17 = nn.BatchNorm2d(k)
        self.norm18 = nn.BatchNorm2d(k)
        self.norm19 = nn.BatchNorm2d(k)
        self.norm20 = nn.BatchNorm2d(k)
        self.norm21 = nn.BatchNorm2d(k)
        self.norm22_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.swish = nn.SiLU()
        
    def forward(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = u1_1_1 + u1_1_2 + u1_2
        # Residual block
        h1 = self.swish(self.norm1(u1))
        h2 = self.swish(self.norm2(self.l2(h1)))
        u3 = self.l3(h2) + u1
        # Residual block
        h3 = self.swish(self.norm3(u3))
        h4 = self.swish(self.norm4(self.l4(h3)))
        u5 = self.l5(h4) + u3
        # Residual block
        h5 = self.swish(self.norm5(u5))
        h6 = self.swish(self.norm6(self.l6(h5)))
        u7 = self.l7(h6) + u5
        # Residual block
        h7 = self.swish(self.norm7(u7))
        h8 = self.swish(self.norm8(self.l8(h7)))
        u9 = self.l9(h8) + u7
        # Residual block
        h9 = self.swish(self.norm9(u9))
        h10 = self.swish(self.norm10(self.l10(h9)))
        u11 = self.l11(h10) + u9
        # Residual block
        h11 = self.swish(self.norm11(u11))
        h12 = self.swish(self.norm12(self.l12(h11)))
        u13 = self.l13(h12) + u11
        # Residual block
        h13 = self.swish(self.norm13(u13))
        h14 = self.swish(self.norm14(self.l14(h13)))
        u15 = self.l15(h14) + u13
        # Residual block
        h15 = self.swish(self.norm15(u15))
        h16 = self.swish(self.norm16(self.l16(h15)))
        u17 = self.l17(h16) + u15
        # Residual block
        h17 = self.swish(self.norm17(u17))
        h18 = self.swish(self.norm18(self.l18(h17)))
        u19 = self.l19(h18) + u17
        # Residual block
        h19 = self.swish(self.norm19(u19))
        h20 = self.swish(self.norm20(self.l20(h19)))
        u21 = self.l21(h20) + u19
        h21 = self.swish(self.norm21(u21))
        # policy network
        h22 = self.l22(h21)
        h22_1 = self.l22_2(torch.flatten(h22, 1))
        # value network
        h22_v = self.swish(self.norm22_v(self.l22_v(h21)))
        h23_v = self.swish(self.l23_v(torch.flatten(h22_v, 1)))
        return h22_1, self.l24_v(h23_v)
    
    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self.swish = nn.SiLU() if memory_efficient else Swish()
