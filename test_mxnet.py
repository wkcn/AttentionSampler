import sys
import numpy as np
import mxnet as mx
import cv2
import mobula
import time
from attention_sampler.attsampler_mx import AttSampler
mobula.op.load('attention_sampler')

tic = time.time()

data = cv2.resize(cv2.imread('imgs/01.jpg'), (224, 224)
                  )[:, :, 0].reshape((1, 1, 224, 224))
map_s = cv2.resize(cv2.imread('imgs/02.jpg'), (224, 224)
                   )[:, :, 0].reshape((1, 224, 224))
ctx = mx.cpu(0)
map_s = mx.nd.array(map_s, ctx=ctx)
data = mx.nd.array(data, ctx=ctx)
map_sx = mx.nd.expand_dims(mx.nd.max(map_s, axis=2), axis=-1)
map_sy = mx.nd.expand_dims(mx.nd.max(map_s, axis=1), axis=-1)
sum_sx = mx.nd.sum(map_sx, axis=(1, 2))
sum_sy = mx.nd.sum(map_sy, axis=(1, 2))
map_sx = mx.nd.broadcast_div(map_sx, mx.nd.reshape(sum_sx, (0, 1, 1)))
map_sy = mx.nd.broadcast_div(map_sy, mx.nd.reshape(sum_sy, (0, 1, 1)))

mx.nd.waitall()


data.attach_grad()
with mx.autograd.record():
    data_pred = AttSampler(scale=0.4375, dense=4)(data, map_sx, map_sy)
    data_pred.backward()
mx.nd.waitall()
print(data_pred.max(), data_pred.min())
print(data.grad.max(), data.grad.min())
print("Okay")
if not hasattr(mx.nd, 'AttSampler'):
    print("TIME", time.time() - tic)
    sys.exit(0)
print("Test consistence")

for _ in range(100):
    data_pred = AttSampler(
        data=data, attx=map_sx, atty=map_sy, scale=0.4375, dense=4)
    data_gt = mx.nd.AttSampler(
        data=data, attx=map_sx, atty=map_sy, scale=0.4375, dense=4)
    mx.nd.waitall()

T = 1000
while 1:
    tic = time.time()
    for _ in range(T):
        data_pred = AttSampler(
            data=data, attx=map_sx, atty=map_sy, scale=0.4375, dense=4)
        data_pred.wait_to_read()
    print("PRED", time.time() - tic)

    tic = time.time()
    for _ in range(T):
        data_pred = mx.nd.AttSampler(
            data=data, attx=map_sx, atty=map_sy, scale=0.4375, dense=4)
        data_pred.wait_to_read()
    print("GT", time.time() - tic)
cv2.imwrite("a.png", data_pred.asnumpy()[0][0])

mobula.testing.assert_almost_equal(data_pred, data_gt)
