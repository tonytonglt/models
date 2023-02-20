import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops

class BBoxAssigner(object):
    __shared__ = ['num_classes', 'assign_on_cpu']
    """
    RCNN targets assignment module

    The assignment consists of three steps:
        1. Match RoIs and ground-truth box, label the RoIs with foreground
           or background sample
        2. Sample anchors to keep the properly ratio between foreground and 
           background
        3. Generate the targets for classification and regression branch

    Args:
        batch_size_per_im (int): Total number of RoIs per image. 
            default 512 
        fg_fraction (float): Fraction of RoIs that is labeled
            foreground, default 0.25
        fg_thresh (float): Minimum overlap required between a RoI
            and ground-truth box for the (roi, gt box) pair to be
            a foreground sample. default 0.5
        bg_thresh (float): Maximum overlap allowed between a RoI
            and ground-truth box for the (roi, gt box) pair to be
            a background sample. default 0.5
        ignore_thresh(float): Threshold for ignoring the is_crowd ground-truth
            if the value is larger than zero.
        use_random (bool): Use random sampling to choose foreground and 
            background boxes, default true
        cascade_iou (list[iou]): The list of overlap to select foreground and
            background of each stage, which is only used In Cascade RCNN.
        num_classes (int): The number of class.
        assign_on_cpu (bool): In case the number of gt box is too large, 
            compute IoU on CPU, default false.
    """

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh=.5,
                 ignore_thresh=-1.,
                 use_random=True,
                 cascade_iou=[0.5, 0.6, 0.7],
                 num_classes=80,
                 assign_on_cpu=False):
        super(BBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.ignore_thresh = ignore_thresh
        self.use_random = use_random
        self.cascade_iou = cascade_iou
        self.num_classes = num_classes
        self.assign_on_cpu = assign_on_cpu

    def __call__(self,
                 rpn_rois,
                 rpn_rois_num,
                 inputs,
                 stage=0,
                 is_cascade=False,
                 add_gt_as_proposals=True):
        gt_classes = inputs['gt_class']
        gt_boxes = inputs['gt_bbox']
        is_crowd = inputs.get('is_crowd', None)
        # rois, tgt_labels, tgt_bboxes, tgt_gt_inds
        # new_rois_num
        outs = self.generate_proposal_target(
            rpn_rois, gt_classes, gt_boxes, self.batch_size_per_im,
            self.fg_fraction, self.fg_thresh, self.bg_thresh, self.num_classes,
            self.ignore_thresh, is_crowd, self.use_random, is_cascade,
            self.cascade_iou[stage], self.assign_on_cpu, add_gt_as_proposals)
        rois = outs[0]
        rois_num = outs[-1]
        # tgt_labels, tgt_bboxes, tgt_gt_inds
        targets = outs[1:4]
        return rois, rois_num, targets

    def generate_proposal_target(self,
                                 rpn_rois,
                                 gt_classes,
                                 gt_boxes,
                                 batch_size_per_im,
                                 fg_fraction,
                                 fg_thresh,
                                 bg_thresh,
                                 num_classes,
                                 ignore_thresh=-1.,
                                 is_crowd=None,
                                 use_random=True,
                                 is_cascade=False,
                                 cascade_iou=0.5,
                                 assign_on_cpu=False,
                                 add_gt_as_proposals=True):

        rois_with_gt = []
        tgt_labels = []
        tgt_bboxes = []
        tgt_gt_inds = []
        new_rois_num = []

        # In cascade rcnn, the threshold for foreground and background
        # is used from cascade_iou
        fg_thresh = cascade_iou if is_cascade else fg_thresh
        bg_thresh = cascade_iou if is_cascade else bg_thresh
        for i, rpn_roi in enumerate(rpn_rois):
            gt_bbox = gt_boxes[i]
            is_crowd_i = is_crowd[i] if is_crowd else None
            gt_class = gt_classes[i].squeeze(-1)

            # Concat RoIs and gt boxes except cascade rcnn or none gt
            if add_gt_as_proposals and gt_bbox.shape[0] > 0:
                bbox = ops.cat([rpn_roi, gt_bbox])
            else:
                bbox = rpn_roi

            # Step1: label bbox
            matches, match_labels = self.label_box(bbox, gt_bbox, fg_thresh, bg_thresh,
                                              False, ignore_thresh, is_crowd_i,
                                              assign_on_cpu)
            # Step2: sample bbox
            sampled_inds, sampled_gt_classes = self.sample_bbox(
                matches, match_labels, gt_class, batch_size_per_im, fg_fraction,
                num_classes, use_random, is_cascade)

            # Step3: make output
            rois_per_image = bbox if is_cascade else bbox.gather(sampled_inds)
            sampled_gt_ind = matches if is_cascade else matches.gather(sampled_inds)

            if gt_bbox.shape[0] > 0:
                sampled_bbox = gt_bbox.gather(sampled_gt_ind)
            else:
                num = rois_per_image.shape[0]
                sampled_bbox = ops.zeros((num, 4), ms.float32)

            rois_per_image = ops.stop_gradient(rois_per_image)
            sampled_gt_ind = ops.stop_gradient(sampled_gt_ind)
            sampled_bbox = ops.stop_gradient(sampled_bbox)

            tgt_labels.append(sampled_gt_classes)
            tgt_bboxes.append(sampled_bbox)
            rois_with_gt.append(rois_per_image)
            tgt_gt_inds.append(sampled_gt_ind)
            new_rois_num.append(sampled_inds.shape[0])
        new_rois_num = ops.cat(new_rois_num)
        return rois_with_gt, tgt_labels, tgt_bboxes, tgt_gt_inds, new_rois_num

    def label_box(self,
                  anchors,
                  gt_boxes,
                  positive_overlap,
                  negative_overlap,
                  allow_low_quality,
                  ignore_thresh,
                  is_crowd=None,
                  assign_on_cpu=False):
        # TODO: finish get_device() / set_device() later
        if assign_on_cpu:
            device = paddle.device.get_device()
            paddle.set_device("cpu")
            iou = self.bbox_overlaps(gt_boxes, anchors)
            paddle.set_device(device)

        else:
            iou = self.bbox_overlaps(gt_boxes, anchors)
        n_gt = gt_boxes.shape[0]
        if n_gt == 0 or is_crowd is None:
            n_gt_crowd = 0
        else:
            n_gt_crowd = ops.nonzero(is_crowd).shape[0]
        if iou.shape[0] == 0 or n_gt_crowd == n_gt:
            # No truth, assign everything to background
            default_matches = ops.full((iou.shape[1],), 0, dtype=ms.int64)
            default_match_labels = ops.full((iou.shape[1],), 0, dtype=ms.int64)
            return default_matches, default_match_labels
        # if ignore_thresh > 0, remove anchor if it is closed to
        # one of the crowded ground-truth
        if n_gt_crowd > 0:
            N_a = anchors.shape[0]
            ones = ops.ones(N_a)
            mask = is_crowd * ones

            if ignore_thresh > 0:
                crowd_iou = iou * mask
                # TODO: finish this later
                valid = (paddle.sum((crowd_iou > ignore_thresh).cast('int32'),
                                    axis=0) > 0).cast('float32')  # paddle
                iou = iou * (1 - valid) - valid

            # ignore the iou between anchor and crowded ground-truth
            iou = iou * (1 - mask) - mask

        matched_vals, matches = iou.topk(k=1, axis=0)
        match_labels = ops.full(matches.shape, -1, dtype=ms.int32)
        # set ignored anchor with iou = -1
        neg_cond = ops.logical_and(matched_vals > -1,
                                   matched_vals < negative_overlap)
        match_labels = ops.where(neg_cond,
                                 ops.zeros_like(match_labels), match_labels)
        match_labels = ops.where(matched_vals >= positive_overlap,
                                 ops.ones_like(match_labels), match_labels)
        if allow_low_quality:
            highest_quality_foreach_gt = iou.max(axis=1, keepdim=True)
            pred_inds_with_highest_quality = ops.logical_and(
                iou > 0, iou == highest_quality_foreach_gt).cast('int32').sum(
                0, keepdim=True)
            match_labels = ops.where(pred_inds_with_highest_quality > 0,
                                     ops.ones_like(match_labels),
                                     match_labels)

        matches = matches.flatten()
        match_labels = match_labels.flatten()

        return matches, match_labels

    def sample_bbox(self,
                    matches,
                    match_labels,
                    gt_classes,
                    batch_size_per_im,
                    fg_fraction,
                    num_classes,
                    use_random=True,
                    is_cascade=False):

        n_gt = gt_classes.shape[0]
        if n_gt == 0:
            # No truth, assign everything to background
            gt_classes = ops.ones(matches.shape, dtype=ms.int32) * num_classes
            # return matches, match_labels + num_classes
        else:
            gt_classes = gt_classes.gather(matches)
            gt_classes = ops.where(match_labels == 0,
                                   ops.ones_like(gt_classes) * num_classes,
                                   gt_classes)
            gt_classes = ops.where(match_labels == -1,
                                      ops.ones_like(gt_classes) * -1, gt_classes)
        if is_cascade:
            index = ops.arange(matches.shape[0])
            return index, gt_classes
        rois_per_image = int(batch_size_per_im)

        fg_inds, bg_inds = self.subsample_labels(gt_classes, rois_per_image, fg_fraction,
                                                 num_classes, use_random)
        if fg_inds.shape[0] == 0 and bg_inds.shape[0] == 0:
            # fake output labeled with -1 when all boxes are neither
            # foreground nor background
            sampled_inds = ops.zeros([1], dtype='int32')
        else:
            sampled_inds = ops.cat([fg_inds, bg_inds])
        sampled_gt_classes = gt_classes.gather(sampled_inds)
        return sampled_inds, sampled_gt_classes

    def subsample_labels(self,
                         labels,
                         num_samples,
                         fg_fraction,
                         bg_label=0,
                         use_random=True):
        positive = ops.nonzero(
            ops.logical_and(labels != -1, labels != bg_label))
        negative = ops.nonzero(labels == bg_label)

        fg_num = int(num_samples * fg_fraction)
        fg_num = min(positive.numel(), fg_num)
        bg_num = num_samples - fg_num
        bg_num = min(negative.numel(), bg_num)
        if fg_num == 0 and bg_num == 0:
            fg_inds = ops.zeros([0], dtype='int32')
            bg_inds = ops.zeros([0], dtype='int32')
            return fg_inds, bg_inds

        # randomly select positive and negative examples

        negative = negative.cast('int32').flatten()
        randperm = ops.Randperm(negative.numel(), dtype=ms.int32)
        bg_perm = randperm(ms.Tensor((negative.numel(),), dtype=ms.int32))
        # TODO: finish slice later
        # bg_perm = paddle.slice(bg_perm, axes=[0], starts=[0], ends=[bg_num])
        bg_perm = ops.slice(bg_perm, (0,), (bg_num,))
        if use_random:
            bg_inds = negative.gather(bg_perm)
        else:
            # bg_inds = paddle.slice(negative, axes=[0], starts=[0], ends=[bg_num])  # TODO: finish slice later
            bg_inds = ops.slice(negative, (0,), (bg_num,))
        if fg_num == 0:
            fg_inds = ops.zeros([0], dtype='int32')
            return fg_inds, bg_inds

        positive = positive.cast('int32').flatten()
        fg_perm = ops.Randperm(positive.numel(), dtype=ms.int32)
        # fg_perm = paddle.slice(fg_perm, axes=[0], starts=[0], ends=[fg_num])  # TODO: finish slice later
        fg_perm = ops.slice(fg_perm, (0,), (fg_num,))
        if use_random:
            fg_inds = positive.gather(fg_perm)
        else:
            # fg_inds = paddle.slice(positive, axes=[0], starts=[0], ends=[fg_num])  # TODO: finish slice later
            fg_inds = ops.slice(positive, (0,), (fg_num,))

        return fg_inds, bg_inds

    def bbox_overlaps(self, boxes1, boxes2):
        """
        Calculate overlaps between boxes1 and boxes2

        Args:
            boxes1 (Tensor): boxes with shape [M, 4]
            boxes2 (Tensor): boxes with shape [N, 4]

        Return:
            overlaps (Tensor): overlaps between boxes1 and boxes2 with shape [M, N]
        """
        M = boxes1.shape[0]
        N = boxes2.shape[0]
        if M * N == 0:
            return ops.zeros((M, N), dtype=ms.float32)
        area1 = self.bbox_area(boxes1)
        area2 = self.bbox_area(boxes2)

        xy_max = ops.minimum(
            ops.unsqueeze(boxes1, 1)[:, :, 2:], boxes2[:, 2:])
        xy_min = ops.maximum(
            ops.unsqueeze(boxes1, 1)[:, :, :2], boxes2[:, :2])
        width_height = xy_max - xy_min
        width_height = width_height.clip(min=0)
        inter = width_height.prod(axis=2)

        overlaps = ops.where(inter > 0, inter /
                             (ops.unsqueeze(area1, 1) + area2 - inter),
                             ops.zeros_like(inter))
        return overlaps

    def bbox_area(self, boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])