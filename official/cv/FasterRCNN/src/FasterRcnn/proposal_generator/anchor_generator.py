import math

import mindspore as ms
from mindspore import ops
from mindspore import nn


class AnchorGenerator(nn.Cell):
    """
    Generate anchors according to the feature maps

    Args:
        anchor_sizes (list[float] | list[list[float]]): The anchor sizes at
            each feature point. list[float] means all feature levels share the
            same sizes. list[list[float]] means the anchor sizes for
            each level. The sizes stand for the scale of input size.
        aspect_ratios (list[float] | list[list[float]]): The aspect ratios at
            each feature point. list[float] means all feature levels share the
            same ratios. list[list[float]] means the aspect ratios for
            each level.
        strides (list[float]): The strides of feature maps which generate
            anchors
        offset (float): The offset of the coordinate of anchors, default 0.

    """

    def __init__(self,
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0],
                 strides=[16.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 offset=0.):
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.variance = variance
        self.cell_anchors = self._calculate_anchors(len(strides))
        self.offset = offset

    def _broadcast_params(self, params, num_features):
        if not isinstance(params[0], (list, tuple)):  # list[float]
            return [params] * num_features
        if len(params) == 1:
            return list(params) * num_features
        return params

    def generate_cell_anchors(self, sizes, aspect_ratios):
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return ms.Tensor(anchors, dtype=ms.float32)

    def _calculate_anchors(self, num_features):
        sizes = self._broadcast_params(self.anchor_sizes, num_features)
        aspect_ratios = self._broadcast_params(self.aspect_ratios, num_features)
        cell_anchors = [
            self.generate_cell_anchors(s, a)
            for s, a in zip(sizes, aspect_ratios)
        ]
        # TODO: figure out what is register_buffer
        # [
        #     self.register_buffer(
        #         t.name, t, persistable=False) for t in cell_anchors
        # ]
        return cell_anchors

    def _create_grid_offsets(self, size, stride, offset):
        grid_height, grid_width = size[0], size[1]
        shifts_x = ops.arange(
            offset * stride, grid_width * stride, step=stride)
        shifts_y = ops.arange(
            offset * stride, grid_height * stride, step=stride)
        shift_y, shift_x = ops.meshgrid((shifts_y, shifts_x), indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        return shift_x, shift_y

    def _grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides,
                                              self.cell_anchors):
            shift_x, shift_y = self._create_grid_offsets(size, stride,
                                                         self.offset)
            shifts = ops.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
            shifts = shifts.reshape(-1, 1, 4)
            base_anchors = base_anchors.reshape(1, -1, 4)

            anchors.append((shifts + base_anchors).reshape(-1, 4))

        return anchors

    def construct(self, input):
        grid_sizes = [feature_map.shape[-2:] for feature_map in input]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return anchors_over_all_feature_maps

    @property
    def num_anchors(self):
        """
        Returns:
            int: number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                For FPN models, `num_anchors` on every feature map is the same.
        """
        return len(self.cell_anchors[0])
