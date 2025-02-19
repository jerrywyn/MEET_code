import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau
        #self.criterion = nn.CrossEntropyLoss()

    def forward(self, logit, target):
    
        #print(logit.shape)
        #print(self.log_cls_num.unsqueeze(0).shape)
    
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        #return self.criterion(logit_adjusted, target)
        return F.cross_entropy(logit_adjusted, target)