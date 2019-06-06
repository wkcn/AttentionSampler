import mobula

@mobula.op.register(need_to_grad=False)
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
        out_size = in_size * self.scale
        threshold = self.scale * self.dense * in_size * 1.0 / att_size
        attx *= out_size
        atty *= out_size
        for j in range(self.iter):
            max_attx = attx.max(1) # (N, 1)
            max_atty = atty.max(1) # (N, 1)
            if j == 0:
                threshold = self.broadcast_minimum(threshold, self.F.maximum(max_attx, max_atty)) # (N, 1)
            else:
                threshold = self.F.maximum(max_attx, max_atty)
            attx = self.broadcast_minimum(threshold, attx)
            atty = self.broadcast_minimum(threshold, atty)
            sum_x = attx.sum(1) # (N, 1)
            sum_y = atty.sum(1) # (N, 1)
            deltax = (out_size - sum_x) / att_size
            deltay = (out_size - sum_y) / att_size
            # compensate the drop value
            attx += deltax
            atty += deltay
        attxi = self.F.empty((N, att_size, 1))
        attxi[0] = 1
        attyi = self.F.empty((N, att_size, 1))
        attyi[0] = 1
        mobula.func.cumsum(attx, attxi, N, att_size)
        mobula.func.cumsum(atty, attyi, N, att_size)
        stepx = attxi[:, -1] / out_size
        stepy = attyi[:, -1] / out_size
        myscale = 2.0 / (att_size - 1)
        index_x = self.F.empty((N, output_size, 1))
        index_y = self.F.empty((N, output_size, 1))
        mobula.func.map_step(attxi, index_x, N, stepx, att_size, out_size)
        mobula.func.map_step(attyi, index_y, N, stepy, att_size, out_size)
        grid = self.F.concat([
            index_x.reshape((N, 1, output_size)).tile((1, output_size, 1)),
            index_y.tile((1, 1, output_size))
        ],dim=1)
        return self.F.BilinearSampler(data, grid)

    def backward(self, dy):
        return [0, 0, 0]
    def infer_shape(self, in_shape):
        dshape = in_shape[0]
        output_size = dshape[2] * self.scale
        return [(dshape[0]. dshape[1], output_size, output_size)]