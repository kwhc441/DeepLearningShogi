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

k = 192
fcl = 256 # fully connected layers
class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand
        self.l2  = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l3  = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l4  = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l5  = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l6  = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l7  = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l8  = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l9  = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
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
        # 追加
        self.l22 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l23 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l24 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l25 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l26 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l27 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l28 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l29 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l30 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l31 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)

        self.l32 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l33 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l34 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l35 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l36 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l37 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l38 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l39 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l40 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l41 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)

        self.l42 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l43 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l44 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l45 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l46 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l47 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l48 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l49 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l50 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l51 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # ここまで追加

        # policy network
        #self.l22 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        #self.l22_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)
        self.l52 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l52_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value network
        #self.l22_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        #self.l23_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        #self.l24_v = nn.Linear(fcl, 1)
        self.l52_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.l53_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l54_v = nn.Linear(fcl, 1)

        self.norm1  = nn.BatchNorm2d(k, eps=2e-05)
        self.norm2  = nn.BatchNorm2d(k, eps=2e-05)
        self.norm3  = nn.BatchNorm2d(k, eps=2e-05)
        self.norm4  = nn.BatchNorm2d(k, eps=2e-05)
        self.norm5  = nn.BatchNorm2d(k, eps=2e-05)
        self.norm6  = nn.BatchNorm2d(k, eps=2e-05)
        self.norm7  = nn.BatchNorm2d(k, eps=2e-05)
        self.norm8  = nn.BatchNorm2d(k, eps=2e-05)
        self.norm9  = nn.BatchNorm2d(k, eps=2e-05)
        self.norm10 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm11 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm12 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm13 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm14 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm15 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm16 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm17 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm18 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm19 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm20 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm21 = nn.BatchNorm2d(k, eps=2e-05)
        #追加
        self.norm22 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm23 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm24 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm25 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm26 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm27 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm28 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm29 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm30 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm31 = nn.BatchNorm2d(k, eps=2e-05)

        self.norm32 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm33 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm34 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm35 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm36 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm37 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm38 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm39 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm40 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm41 = nn.BatchNorm2d(k, eps=2e-05)

        self.norm42 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm43 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm44 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm45 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm46 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm47 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm48 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm49 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm50 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm51 = nn.BatchNorm2d(k, eps=2e-05)
        #ここまで追加
        #self.norm22_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM, eps=2e-05)
        self.norm52_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM, eps=2e-05)

    def forward(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = u1_1_1 + u1_1_2 + u1_2
        # Residual block 01
        h1 = F.relu(self.norm1(u1))
        h2 = F.relu(self.norm2(self.l2(h1)))
        u3 = self.l3(h2) + u1
        # Residual block 02
        h3 = F.relu(self.norm3(u3))
        h4 = F.relu(self.norm4(self.l4(h3)))
        u5 = self.l5(h4) + u3
        # Residual block 03
        h5 = F.relu(self.norm5(u5))
        h6 = F.relu(self.norm6(self.l6(h5)))
        u7 = self.l7(h6) + u5
        # Residual block 04
        h7 = F.relu(self.norm7(u7))
        h8 = F.relu(self.norm8(self.l8(h7)))
        u9 = self.l9(h8) + u7
        # Residual block 05
        h9 = F.relu(self.norm9(u9))
        h10 = F.relu(self.norm10(self.l10(h9)))
        u11 = self.l11(h10) + u9
        # Residual block 06
        h11 = F.relu(self.norm11(u11))
        h12 = F.relu(self.norm12(self.l12(h11)))
        u13 = self.l13(h12) + u11
        # Residual block 07
        h13 = F.relu(self.norm13(u13))
        h14 = F.relu(self.norm14(self.l14(h13)))
        u15 = self.l15(h14) + u13
        # Residual block 08
        h15 = F.relu(self.norm15(u15))
        h16 = F.relu(self.norm16(self.l16(h15)))
        u17 = self.l17(h16) + u15
        # Residual block 09
        h17 = F.relu(self.norm17(u17))
        h18 = F.relu(self.norm18(self.l18(h17)))
        u19 = self.l19(h18) + u17
        # Residual block 10
        h19 = F.relu(self.norm19(u19))
        h20 = F.relu(self.norm20(self.l20(h19)))
        u21 = self.l21(h20) + u19
        #h21 = F.relu(self.norm21(u21))

        #追加
        # Residual block 11
        h21 = F.relu(self.norm21(u21))
        h22 = F.relu(self.norm22(self.l22(h21)))
        u23 = self.l23(h22) + u21
        # Residual block 12
        h23 = F.relu(self.norm23(u23))
        h24 = F.relu(self.norm24(self.l24(h23)))
        u25 = self.l25(h24) + u23
        # Residual block 13
        h25 = F.relu(self.norm25(u25))
        h26 = F.relu(self.norm26(self.l26(h25)))
        u27 = self.l27(h26) + u25
        # Residual block 14
        h27 = F.relu(self.norm27(u27))
        h28 = F.relu(self.norm28(self.l28(h27)))
        u29 = self.l29(h28) + u27
        # Residual block 15
        h29 = F.relu(self.norm29(u29))
        h30 = F.relu(self.norm30(self.l30(h29)))
        u31 = self.l31(h30) + u29
        # Residual block 16
        h31 = F.relu(self.norm31(u31))
        h32 = F.relu(self.norm32(self.l32(h31)))
        u33 = self.l33(h32) + u31
        # Residual block 17
        h33 = F.relu(self.norm33(u33))
        h34 = F.relu(self.norm34(self.l34(h33)))
        u35 = self.l35(h34) + u33
        # Residual block 18
        h35 = F.relu(self.norm35(u35))
        h36 = F.relu(self.norm36(self.l36(h35)))
        u37 = self.l37(h36) + u35
        # Residual block 19
        h37 = F.relu(self.norm37(u37))
        h38 = F.relu(self.norm38(self.l38(h37)))
        u39 = self.l39(h38) + u37
        # Residual block 20
        h39 = F.relu(self.norm39(u39))
        h40 = F.relu(self.norm40(self.l40(h39)))
        u41 = self.l41(h40) + u39

        # Residual block 21
        h41 = F.relu(self.norm41(u41))
        h42 = F.relu(self.norm42(self.l42(h41)))
        u43 = self.l43(h42) + u41
        # Residual block 22
        h43 = F.relu(self.norm43(u43))
        h44 = F.relu(self.norm44(self.l44(h43)))
        u45 = self.l45(h44) + u43
        # Residual block 23
        h45 = F.relu(self.norm45(u45))
        h46 = F.relu(self.norm46(self.l46(h45)))
        u47 = self.l47(h46) + u45
        # Residual block 24
        h47 = F.relu(self.norm47(u47))
        h48 = F.relu(self.norm48(self.l48(h47)))
        u49 = self.l49(h48) + u47
        # Residual block 25
        h49 = F.relu(self.norm49(u49))
        h50 = F.relu(self.norm50(self.l50(h49)))
        u51 = self.l51(h50) + u49
        h51 = F.relu(self.norm51(u51))
        #ここまで追加

        # policy network
        #h22 = self.l22(h21)
        #h22_1 = self.l22_2(torch.flatten(h22, 1))
        h52 = self.l52(h51)
        h52_1 = self.l52_2(torch.flatten(h52, 1))
        # value network
        #h22_v = F.relu(self.norm22_v(self.l22_v(h21)))
        #h23_v = F.relu(self.l23_v(torch.flatten(h22_v, 1)))
        #return h22_1, self.l24_v(h23_v)
        h52_v = F.relu(self.norm52_v(self.l52_v(h51)))
        h53_v = F.relu(self.l53_v(torch.flatten(h52_v, 1)))
        return h52_1, self.l54_v(h53_v)
