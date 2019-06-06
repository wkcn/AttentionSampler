import numpy as np
import mxnet as mx
import cv2
import mobula
mobula.op.load('attention_sampler')

map_s = cv2.resize(cv2.imread('imgs/01.jpg'),(224,224))[:,:,0].reshape((1,224,224))
data = cv2.resize(cv2.imread('imgs/02.jpg'),(224,224))[:,:,0].reshape((1,1,224,224))
ctx= mx.cpu(0)
map_s = mx.nd.array(map_s,ctx=ctx)
data = mx.nd.array(data,ctx=ctx)
map_sx = mx.nd.expand_dims(mx.nd.max(map_s, axis=2), axis=-1)
map_sy = mx.nd.expand_dims(mx.nd.max(map_s, axis=1), axis=-1)
sum_sx = mx.nd.sum(map_sx, axis=(1,2))
sum_sy = mx.nd.sum(map_sy, axis=(1,2))
map_sx = mx.nd.broadcast_div(map_sx, mx.nd.reshape(sum_sx,(0,1,1)))
map_sy = mx.nd.broadcast_div(map_sy, mx.nd.reshape(sum_sy,(0,1,1)))

data_s = mobula.op.AttSampler(data=data, attx=map_sx, atty=map_sy, scale=0.4375, dense=4)
mx.nd.waitall()
print("TEST")
