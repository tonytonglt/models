import numpy as np
import mindspore as ms
from mindspore import ops

from bbox_utils import nonempty_bbox


class BBoxPostProcess(object):
    def __init__(self, num_classes=80, decode=None, nms=None):
        super(BBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms

    def __call__(self, head_out, rois, im_shape, scale_factor):
        """
        Decode the bbox and do NMS if needed.

        Args:
            head_out (tuple): bbox_pred and cls_prob of bbox_head output.
            rois (tuple): roi and rois_num of rpn_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
        """
        if self.nms is not None:
            bboxes, score = self.decode(head_out, rois, im_shape, scale_factor)
            bboxes = bboxes.reshape(bboxes.shape[0], -1)
            bbox_pred, bbox_num, _ = self.nms(bboxes, score, 0, nms_cfg={}, return_inds=True)

        else:
            bbox_pred, bbox_num = self.decode(head_out, rois, im_shape,
                                              scale_factor)

        # if self.export_onnx:
        #     # add fake box after postprocess when exporting onnx
        #     fake_bboxes = ms.Tensor(
        #         np.array(
        #             [[0., 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=ms.float32))
        #
        #     bbox_pred = ops.cat([bbox_pred, fake_bboxes])
        #     bbox_num = bbox_num + 1

        return bbox_pred, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to
        get final prediction.

        Notes:
        Currently only support bs = 1.

        Args:
            bboxes (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            pred_result (Tensor): The final prediction results with shape [N, 6]
                including labels, scores and bboxes.
        """
        bboxes_list = []
        bbox_num_list = []
        id_start = 0
        fake_bboxes = ms.Tensor(
            np.array(
                [[0., 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=ms.float32))
        fake_bbox_num = ms.Tensor(np.array([1], dtype=ms.int32))

        # add fake bbox when output is empty for each batch
        for i in range(bbox_num.shape[0]):
            if bbox_num[i] == 0:
                bboxes_i = fake_bboxes
                bbox_num_i = fake_bbox_num
            else:
                bboxes_i = bboxes[id_start:id_start + bbox_num[i], :]
                bbox_num_i = bbox_num[i]
                id_start += bbox_num[i]
            bboxes_list.append(bboxes_i)
            bbox_num_list.append(bbox_num_i)
        bboxes = ops.cat(bboxes_list)
        bbox_num = ops.cat(bbox_num_list)

        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)

        origin_shape_list = []
        scale_factor_list = []
        # scale_factor: scale_y, scale_x
        for i in range(bbox_num.shape[0]):
            expand_shape = paddle.expand(origin_shape[i:i + 1, :],
                                         [bbox_num[i], 2])
            scale_y, scale_x = scale_factor[i][0], scale_factor[i][1]
            scale = ops.cat([scale_x, scale_y, scale_x, scale_y])
            expand_scale = paddle.expand(scale, [bbox_num[i], 4])
            origin_shape_list.append(expand_shape)
            scale_factor_list.append(expand_scale)

        self.origin_shape_list = ops.cat(origin_shape_list)
        scale_factor_list = ops.cat(scale_factor_list)

        # bboxes: [N, 6], label, score, bbox
        pred_label = bboxes[:, 0:1]
        pred_score = bboxes[:, 1:2]
        pred_bbox = bboxes[:, 2:]
        # rescale bbox to original image
        scaled_bbox = pred_bbox / scale_factor_list
        origin_h = self.origin_shape_list[:, 0]
        origin_w = self.origin_shape_list[:, 1]
        zeros = ops.zeros_like(origin_h)
        # clip bbox to [0, original_size]
        x1 = ops.maximum(ops.minimum(scaled_bbox[:, 0], origin_w), zeros)
        y1 = ops.maximum(ops.minimum(scaled_bbox[:, 1], origin_h), zeros)
        x2 = ops.maximum(ops.minimum(scaled_bbox[:, 2], origin_w), zeros)
        y2 = ops.maximum(ops.minimum(scaled_bbox[:, 3], origin_h), zeros)
        pred_bbox = ops.stack([x1, y1, x2, y2], axis=-1)
        # filter empty bbox
        keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
        keep_mask = ops.unsqueeze(keep_mask, dim=1)
        pred_label = ops.where(keep_mask, pred_label,
                               ops.ones_like(pred_label) * -1)
        pred_result = ops.cat([pred_label, pred_score, pred_bbox], axis=1)
        return bboxes, pred_result, bbox_num