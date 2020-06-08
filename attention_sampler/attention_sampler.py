import mobula


@mobula.op.register
class AttSamplerGrid:
    def __init__(self, scale=1.0, dense=4, iters=5):
        self.scale = scale
        self.dense = dense
        self.iters = iters

    def forward(self, data, attx, atty):
        F = self.F._mobula_hack
        # attx: (N, W, 1)
        # atty: (N, H, 1)
        N, _, in_size, _ = data.shape
        att_size = attx.shape[1]
        out_size = int(in_size * self.scale)
        threshold = float(self.scale * self.dense * in_size) / att_size
        attx = attx * out_size
        atty = atty * out_size
        for j in range(self.iters):
            max_attx = F.max(attx, 1, keepdims=True)  # (N, 1, 1)
            max_atty = F.max(atty, 1, keepdims=True)  # (N, 1, 1)
            if j == 0:
                threshold = F.minimum(F.minimum(
                    max_attx, max_atty), threshold)  # (N, 1, 1)
            else:
                threshold = F.minimum(max_attx, max_atty)
            F.broadcast_minimum(threshold, attx, out=attx)
            F.broadcast_minimum(threshold, atty, out=atty)
            sum_x = F.sum(attx, 1, keepdims=True)  # (N, 1, 1)
            sum_y = F.sum(atty, 1, keepdims=True)  # (N, 1, 1)
            deltax = (out_size - sum_x) / att_size
            deltay = (out_size - sum_y) / att_size
            # compensate the drop value
            attx += deltax
            atty += deltay
        '''
        it is the same as the original implemenation.
        the first element is 1.
        '''
        attx[:, 0] = 1
        atty[:, 0] = 1
        if hasattr(F, 'cumsum'):
            attxi = F.cumsum(attx, 1)
            attyi = F.cumsum(atty, 1)
        else:
            attxi = self.F.empty((N, att_size, 1))
            attyi = self.F.empty((N, att_size, 1))
            attxi[:, 0] = 1
            attyi[:, 0] = 1
            mobula.func.cumsum(N, attx, attxi, att_size)
            mobula.func.cumsum(N, atty, attyi, att_size)
        stepx = attxi[:, -1] / out_size
        stepy = attyi[:, -1] / out_size
        ctx = F.get_ctx(stepx)
        index_x = F.empty((N, out_size, 1), ctx=ctx)
        index_y = F.empty((N, out_size, 1), ctx=ctx)
        # note: x and y are inverse.
        mobula.func.map_step(N, attxi, index_y, stepx, att_size, out_size)
        mobula.func.map_step(N, attyi, index_x, stepy, att_size, out_size)
        return F.tile(F.reshape(index_x, (N, 1, out_size)), (1, out_size, 1)),\
            F.tile(index_y, (1, 1, out_size))

    def backward(self, dy_x, dy_y):
        return [0, 0, 0]

    def infer_shape(self, in_shape):
        dshape = in_shape[0]
        out_size = int(dshape[2] * self.scale)
        oshape = (dshape[0], out_size, out_size)
        return in_shape, [oshape, oshape]
