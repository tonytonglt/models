import math

import mindspore as ms
from mindspore import ops


class RCNNBox(object):
    def __init__(self,
                 prior_box_var=[10., 10., 5., 5.],
                 code_type="decode_center_size",
                 box_normalized=False,
                 num_classes=80,
                 export_onnx=False):
        super(RCNNBox, self).__init__()
        self.prior_box_var = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.num_classes = num_classes

    def __call__(self, bbox_head_out, rois, im_shape, scale_factor):
        bbox_pred = bbox_head_out[0]
        cls_prob = bbox_head_out[1]
        roi = rois[0]
        rois_num = rois[1]

        origin_shape_list = []
        if isinstance(roi, list):
            batch_size = len(roi)
        else:
            # batch_size = paddle.slice(paddle.shape(im_shape), [0], [0], [1])
            batch_size = ops.slice(ms.Tensor(im_shape.shape), (0,), (1,))

        # bbox_pred.shape: [N, C*4]
        for idx in range(batch_size):
            rois_num_per_im = rois_num[idx]
            expand_im_shape = ops.expand(im_shape[idx, :],
                                         ms.Tensor((rois_num_per_im.asnumpy().item(), 2)))
            origin_shape_list.append(expand_im_shape)

        origin_shape = ops.concat(origin_shape_list)

        # bbox_pred.shape: [N, C*4]
        # C=num_classes in faster/mask rcnn(bbox_head), C=1 in cascade rcnn(cascade_head)
        bbox = ops.concat(roi)
        bbox = delta2bbox(bbox_pred, bbox, self.prior_box_var)
        scores = cls_prob[:, :-1]

        # bbox.shape: [N, C, 4]
        # bbox.shape[1] must be equal to scores.shape[1]
        total_num = bbox.shape[0]
        bbox_dim = bbox.shape[-1]
        bbox = bbox.expand(ms.Tensor([total_num, self.num_classes, bbox_dim]))

        origin_h = origin_shape[:, 0].unsqueeze(dim=1)
        origin_w = origin_shape[:, 1].unsqueeze(dim=1)
        zeros = ops.zeros_like(origin_h)
        x1 = ops.maximum(ops.minimum(bbox[:, :, 0], origin_w), zeros)
        y1 = ops.maximum(ops.minimum(bbox[:, :, 1], origin_h), zeros)
        x2 = ops.maximum(ops.minimum(bbox[:, :, 2], origin_w), zeros)
        y2 = ops.maximum(ops.minimum(bbox[:, :, 3], origin_h), zeros)
        bboxes = ops.stack([x1, y1, x2, y2], axis=-1)
        # bboxes = (bbox, rois_num)
        return bboxes, scores


def bbox2delta(src_boxes, tgt_boxes, weights):
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * ops.log(tgt_w / src_w)
    dh = wh * ops.log(tgt_h / src_h)

    deltas = ops.stack((dx, dy, dw, dh), axis=1)
    return deltas


def delta2bbox(deltas, boxes, weights):
    clip_scale = math.log(1000.0 / 16)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    # Prevent sending too large values into paddle.exp()
    dw = dw.clip(xmax=clip_scale, xmin=None)
    dh = dh.clip(xmax=clip_scale, xmin=None)

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = ops.exp(dw) * widths.unsqueeze(1)
    pred_h = ops.exp(dh) * heights.unsqueeze(1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = ops.stack(pred_boxes, axis=-1)

    return pred_boxes


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr=.05,
                   nms_cfg=None,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """
    Refers to
    https://github.com/open-mmlab/mmdetection/blob/d81990802cbd80f6d49fc35cdd255ebc353e4541/mmdet/core/post_processing/bbox_nms.py
    """

    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): a dict that contains the arguments of nms operations
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.
    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.shape[1]
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.reshape(multi_scores.shape[0], -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(ms.Tensor(multi_scores.size(0), num_classes, 4))

    scores = multi_scores

    labels = ops.arange(num_classes, dtype=ms.int64)
    labels = labels.reshape(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.reshape(-1, 1).expand(ms.Tensor(multi_scores.size(0), num_classes))
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero().squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        dets = ops.concat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)  #修改batched_nms

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    """transfer det to [label_id, score, x1, y1, x2, y2]"""
    labels = labels[keep].reshape(-1, 1)
    labels = ms.Tensor(labels, dtype=ms.float32)  # type casting to float32
    dets = ops.concat((labels, dets), axis=1)
    if return_inds:
        return dets, dets.shape[0], inds[keep]
    else:
        return dets, dets.shape[0]


def batched_nms(boxes,
                scores,
                idxs,
                nms_cfg,
                class_agnostic=False,
                nms_threshold=.5,
                keep_top_k=100,
                nms_top_k=-1):
    """
    Refers to
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
    TODO: Is it possible to use mindspore.ops.NMSWithMask???
    https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.NMSWithMask.html#mindspore.ops.NMSWithMask
    """

    """
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    nms_cfg: Optional[Dict],
    class_agnostic: bool = False) -> Tuple[Tensor, Tensor]
    """

    r"""Performs non-maximum suppression in a batched fashion.
    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.
    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.
            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.
    Returns:
        tuple: kept dets and indice.
        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return ops.concat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.shape[-1] == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes.dtype) * (
                max_coordinate + ms.Tensor(1).to(boxes.dtype))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = ops.concat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                    axis=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes.dtype) * (
                max_coordinate + ms.Tensor(1).to(boxes.dtype))
            boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    # nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)  # ??? TODO: figure out what is nms_op
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = keep_top_k
        total_mask = scores.new_zeros(scores.shape, dtype=ms.bool_)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.shape)
        unique_idxs, _ = ops.unique(idxs)
        for id in unique_idxs:
            mask = (idxs == id).nonzero().reshape(-1)
            nms_op = ops.NMSWithMask(iou_threshold=nms_threshold)
            bbox_with_scores = ops.concat((boxes_for_nms[mask], scores[mask].reshape(-1, 1)), axis=1)
            ordered_box, ordered_idx, valid_mask = nms_op(bbox_with_scores)
            # dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            keep = ms.Tensor(ordered_idx.asnumpy()[valid_mask.asnumpy()], dtype=ms.int32)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = ops.gather(scores[mask], keep, axis=0)
            # scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero().reshape(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = ops.concat([scores[:, None], boxes], -1)
    return boxes, keep


def nonempty_bbox(boxes, min_size=0, return_mask=False):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = ops.logical_and(h > min_size, w > min_size)
    if return_mask:
        return mask
    keep = ops.nonzero(mask).flatten()
    return keep

