import sys
import numpy as np
import torch
import cv2
import mobula
import time
from attention_sampler.attsampler_th import AttSampler
mobula.op.load('attention_sampler')

tic = time.time()
data = cv2.resize(cv2.imread('imgs/01.jpg'), (224, 224)
                  )[:, :, 0].reshape((1, 1, 224, 224))
map_s = cv2.resize(cv2.imread('imgs/02.jpg'), (224, 224)
                   )[:, :, 0].reshape((1, 224, 224))
device = torch.device('cpu')
map_s = torch.tensor(map_s, dtype=torch.float32, device=device)
data = torch.tensor(data, dtype=torch.float32,
                    requires_grad=True, device=device)
map_sx = torch.unsqueeze(torch.max(map_s, 2)[0], dim=2)
map_sy = torch.unsqueeze(torch.max(map_s, 1)[0], dim=2)
sum_sx = torch.sum(map_sx, dim=(1, 2), keepdim=True)
sum_sy = torch.sum(map_sy, dim=(1, 2), keepdim=True)
map_sx /= sum_sx
map_sy /= sum_sy

data_pred = AttSampler(scale=0.4375, dense=4)(data, map_sx, map_sy)
data_pred.sum().backward()
print(data_pred.max(), data_pred.min())
print(data.grad.max(), data.grad.min())
print("TIME", time.time() - tic)
print("Okay")
