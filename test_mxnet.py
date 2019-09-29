import sys
import numpy as np
import mxnet as mx
import cv2
import mobula
import time
from attention_sampler import attsampler_mx
mobula.op.load('attention_sampler')


tic = time.time()
data = cv2.resize(cv2.imread('imgs/01.jpg'), (224, 224)
                  )[:, :, 0].reshape((1, 1, 224, 224))
map_s = cv2.resize(cv2.imread('imgs/02.jpg'), (224, 224)
                   )[:, :, 0].reshape((1, 224, 224))
ctx = mx.cpu(0)
map_s = mx.nd.array(map_s, ctx=ctx)
data = mx.nd.array(data, ctx=ctx)
data.attach_grad()
map_sx = mx.nd.expand_dims(mx.nd.max(map_s, axis=2), axis=-1)
map_sy = mx.nd.expand_dims(mx.nd.max(map_s, axis=1), axis=-1)
sum_sx = mx.nd.sum(map_sx, axis=(1, 2), keepdims=True)
sum_sy = mx.nd.sum(map_sy, axis=(1, 2), keepdims=True)
map_sx /= sum_sx
map_sy /= sum_sy

mx.nd.waitall()


with mx.autograd.record():
    data_pred = mx.nd.contrib.AttSampler(data=data,
                                         attx=map_sx, atty=map_sy,
                                         scale=0.4375, dense=4)
    data_pred.sum().backward()
mx.nd.waitall()
print(data_pred.max(), data_pred.min())
print(data.grad.max(), data.grad.min())
print("TIME", time.time() - tic)
print("Okay")
