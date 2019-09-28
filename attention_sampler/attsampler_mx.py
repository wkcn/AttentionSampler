import mobula
import mxnet as mx


class AttSampler(mx.gluon.HybridBlock):
    def __init__(self, scale=1.0, dense=4, iters=5):
        super(AttSampler, self).__init__()
        self.scale = scale
        self.dense = dense
        self.iters = iters

    def hybrid_forward(self, F, data, attx, atty):
        grid = mobula.op.AttSamplerGrid(data, attx, atty,
                                        scale=self.scale,
                                        dense=self.dense,
                                        iters=self.iters)
        return F.BilinearSampler(data, grid)
