import numpy as np
import paddle
from paddle.nn.initializer import Normal

"""testing ParamAttr"""
weight_attr = paddle.ParamAttr(initializer=Normal(
                mean=0.0, std=0.01))

"""testing nonzero"""
x0 = paddle.to_tensor([[-1.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0],
                       [0.0, 0.0, 3.0]], dtype='float32')
out_z1 = paddle.nonzero(x0 >= 0)

"""testing zeros"""
zeros = paddle.zeros([1], dtype='float32')

"""testing cast"""
x1 = paddle.to_tensor([[-1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 3.0]], dtype='float32')
casted = x1.cast('int64')

"""testing reshape"""
x2 = paddle.to_tensor([[-1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 3.0]], dtype='float32')
reshaped = paddle.reshape(x2, shape=(-1, ))

"""testing exp"""
x3 = paddle.to_tensor([[-1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 3.0]], dtype='float32')
exp_x3 = paddle.exp(x3)

"""testing slice, not right"""
x4 = paddle.to_tensor([[-1.0, 0.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0, 0.0],
                      [0.0, 0.0, 3.0, 0.0],
                      [0.0, 0.0, 0.0, 4.0]], dtype='float32')
d1 = paddle.slice(x4, axes=[2], starts=[0], ends=[1])
d2 = paddle.slice(x4, axes=[2], starts=[1], ends=[2])
d3 = paddle.slice(x4, axes=[2], starts=[2], ends=[3])
d4 = paddle.slice(x4, axes=[2], starts=[3], ends=[4])
# x4 = paddle.to_tensor([[[1, 1, 1], [2, 2, 2]],
#                       [[3, 3, 3], [4, 4, 4]],
#                       [[5, 5, 5], [6, 6, 6]]], dtype='float32')
# d1 = paddle.slice(x4, axes=[0, 0, 0], starts=[0, 0, 0], ends=[0, 0, 2])
# d2 = paddle.slice(x4, axes=[0], starts=[1], ends=[2])
# d3 = paddle.slice(x4, axes=[0], starts=[2], ends=[3])
# d4 = paddle.slice(x4, axes=[0], starts=[3], ends=[4])

deltas = paddle.to_tensor([[-1.0, 0.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0, 0.0],
                           [0.0, 0.0, 3.0, 0.0],
                           [0.0, 0.0, 0.0, 4.0],
                           [0.0, 0.0, 0.0, 5.0]], dtype='float32')
# deltas = deltas.reshape((0, -1, 4))
dx = paddle.slice(deltas, axes=[2], starts=[0], ends=[1])
dy = paddle.slice(deltas, axes=[2], starts=[1], ends=[2])
dw = paddle.slice(deltas, axes=[2], starts=[2], ends=[3])
dh = paddle.slice(deltas, axes=[2], starts=[3], ends=[4])

"""testing slice"""
bg_perm = paddle.to_tensor([9, 3, 15, 11, 6, 4, 13, 14, 12, 10, 8, 1, 5, 0, 2, 7], dtype='int32')
bg_perm = paddle.slice(bg_perm, axes=[0], starts=[0], ends=[16])

"""testing numpy to tensor"""
numpy_arr = np.eye(4)
paddle_tensor = paddle.to_tensor(numpy_arr, dtype='float32')
shape = paddle.shape(paddle_tensor)

"""testing meshgrid"""
x = paddle.to_tensor([1, 2, 3, 4], dtype='float32')
y = paddle.to_tensor([5, 6, 7], dtype='float32')
grid_x, grid_y = paddle.meshgrid(x, y)

print()


# import paddle
# # 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
#
# def pytorch_params(pth_file):
#     par_dict = paddle.load(pth_file)
#     pt_params = {}
#     for name in par_dict:
#         parameter = par_dict[name]
#         print(name, parameter.numpy().shape)
#         pt_params[name] = parameter.numpy()
#     return pt_params
#
# if __name__ == '__main__':
#     pt_params = pytorch_params('C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.pdparams')
#     print()


