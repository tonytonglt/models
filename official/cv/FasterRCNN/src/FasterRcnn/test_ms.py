import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal

in_channels = 2048
num_class = 80
bbox_delta = nn.Dense(in_channels, num_class + 1, weight_init=Normal(0.02))
bbox_delta_new = nn.Dense(in_channels, num_class + 1, weight_init='normal')

"""testing Tensor.squeeze(axis)"""
ones = ops.ones((1000, 2048, 1, 1), ms.float32)
ones = ones.squeeze(axis=(2, 3))

"""testing Tensor.nonzero()"""
x0 = ms.Tensor(np.array([[-1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0]]), ms.int32)
out_z1 = ops.nonzero(x0 >= 0)

"""testing zeros"""
zeros = ops.zeros(1, ms.float32)

"""testing cast"""
x1 = ms.Tensor(np.array([[-1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0]]), ms.float32)
casted = ops.cast(x1, ms.int64)
# casted = x1.cast(ms.int64) # wrong way

"""testing reshape"""
x2 = ms.Tensor(np.array([[-1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0]]), ms.float32)
reshaped = x2.reshape((-1,))
reshaped2 = x2.reshape(-1, 9)
reshaped3 = x2.reshape(-1)
"""testing exp"""
x3 = ms.Tensor(np.array([[-1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0]]), ms.float32)
exp_x3 = x3.exp()

"""testing slice, not right"""
deltas = ms.Tensor(np.array([[-1.0, 0.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0, 0.0],
                             [0.0, 0.0, 3.0, 0.0],
                             [0.0, 0.0, 0.0, 4.0],
                             [0.0, 0.0, 0.0, 5.0]]).astype(np.int32))
d1 = ops.slice(deltas, (0, 0), (len(deltas), 1))
d2 = ops.slice(deltas, (0, 1), (len(deltas), 1))
d3 = ops.slice(deltas, (0, 2), (len(deltas), 1))
d4 = ops.slice(deltas, (0, 3), (len(deltas), 1))

# deltas = ops.reshape(deltas, (0, -1, 4))

# d1 = ops.slice(deltas, (0, 0), (5, 4))
# d2 = ops.slice(deltas, (1, 0), (1, 4))
# d3 = ops.slice(deltas, (2, 0), (1, 4))
# d4 = ops.slice(deltas, (3, 0), (1, 4))
# x4 = ms.Tensor(np.array([[[1, 1, 1], [2, 2, 2]],
#                          [[3, 3, 3], [4, 4, 4]],
#                          [[5, 5, 5], [6, 6, 6]]]).astype(np.int32))
# sliced_1 = ops.slice(x4, (0, 0, 0), (1, 1, 3))
# sliced_2 = ops.slice(x4, (1, 0, 0), (1, 1, 3))
# sliced_3 = ops.slice(x4, (2, 0, 0), (1, 1, 3))

image_shape = (8, 2)
batch_size = ops.slice(ms.Tensor(np.array([[500, 500]])), (0, 0), (1, 1))


"""testing expand"""
input_x = ms.Tensor(np.array([[1], [2], [3]]), ms.float32)
size = ms.Tensor(np.array([3, 4]), ms.int32)
y1 = ops.expand(input_x, size)
y2 = input_x.expand(ms.Tensor((3, 4)))  # size must be tensor
y3 = ops.expand(input_x, ms.Tensor((3, 4)))

"""testing sort/Sort"""
before_sort = ops.ones((1000, 80), ms.float32)
before_sort[0][-1] = 0
before_sort[-1][-2] = 0.01
after_sort, inds = before_sort.sort(descending=True)

"""testing to"""
tmp = ops.ones(1, ms.float32)
before_to = ops.ones(1, ms.int32)
after_to = before_to.to(tmp.dtype)

"""testing new_zeros & dtype=bool"""
dtype_bool_tmp = ops.ones((1000, 80), ms.float32)
dtype_bool = dtype_bool_tmp.new_zeros(dtype_bool_tmp.shape, dtype=ms.bool_)

"""testing randperm & slice"""
# randperm = ops.Randperm(16, dtype=ms.int32)  # Unsupported op [Randperm] on CPU
# bg_perm = randperm(ms.Tensor((16, )))

bg_perm = ms.Tensor((9, 3, 15, 11, 6, 4, 13, 14, 12, 10, 8, 1, 5, 0, 2, 7), dtype=ms.int32)
bg_perm = ops.slice(bg_perm, (0,), (16,))

"""testing numpy to tensor"""
numpy_arr = np.eye(4)
ms_tensor = ms.Tensor(numpy_arr, dtype=ms.float32)
shape = ms_tensor.shape

"""testing meshgrid"""
x = ms.Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
y = ms.Tensor(np.array([5, 6, 7]).astype(np.int32))
# z = ms.Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
grid_x, grid_y = ops.meshgrid(x, y, indexing='ij')


print()
