import numpy as np 

#import torch
#import matplotlib.pyplot as plt
#b = torch.tensor([1,2,3])
#b = np.array([1,2,3])
#print(b)
print('junsun')
def ac(a, b):
    return a * b

c = ac(5, 6)
print(c)

import torch
import torch_xla.core.xla_model as xm

dev = xm.xla_device()
t1 = torch.randn(3,3,device=dev)
t2 = torch.randn(3,3,device=dev)
print(t1 + t2)
print(t1)
