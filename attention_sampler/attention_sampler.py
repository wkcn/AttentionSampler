import mobula

@mobula.op.register
class AttSampler:
    def __init__(self, scale=1.0, dense=4, iters=5):
        self.scale = scale
        self.dense = dense
        self.iters = iters
    def forward(self, data, attx, atty):
        # attx: (N, W, 1)
        # atty: (N, H, 1)
        N, _, in_size, _ = data.shape
        att_size = attx.shape[1]
        out_size = int(in_size * self.scale)
        threshold = float(self.scale * self.dense * in_size) / att_size
        attx = attx * out_size
        atty = atty * out_size
        for j in range(self.iters):
            max_attx = attx.max(1) # (N, 1)
            max_atty = atty.max(1) # (N, 1)
            if j == 0:
                threshold = self.F.minimum(self.F.minimum(max_attx, max_atty), threshold) # (N, 1)
            else:
                threshold = self.F.minimum(max_attx, max_atty)
            self.F.broadcast_minimum(threshold, attx, out=attx)
            self.F.broadcast_minimum(threshold, atty, out=atty)
            sum_x = attx.sum(1) # (N, 1)
            sum_y = atty.sum(1) # (N, 1)
            deltax = (out_size - sum_x) / att_size
            deltay = (out_size - sum_y) / att_size
            # compensate the drop value
            attx += deltax
            atty += deltay
        attxi = self.F.ones((N, att_size, 1))
        attyi = self.F.ones((N, att_size, 1))
        mobula.func.cumsum(N, attx, attxi, att_size)
        mobula.func.cumsum(N, atty, attyi, att_size)
        stepx = attxi[:, -1] / out_size
        stepy = attyi[:, -1] / out_size
        index_x = self.F.empty((N, out_size, 1))
        index_y = self.F.empty((N, out_size, 1))
        mobula.func.map_step(N, attxi, index_y, stepx, att_size, out_size)
        mobula.func.map_step(N, attyi, index_x, stepy, att_size, out_size)
        grid = self.F.stack(
            index_x.reshape((N, 1, out_size)).tile((1, out_size, 1)),
            index_y.tile((1, 1, out_size)), axis=1)
        return self.F.BilinearSampler(data, grid)

    def backward(self, dy):
        return [0, 0, 0]
    def infer_shape(self, in_shape):
        dshape = in_shape[0]
        out_size = int(dshape[2] * self.scale)
        return in_shape, [(dshape[0], dshape[1], out_size, out_size)]
