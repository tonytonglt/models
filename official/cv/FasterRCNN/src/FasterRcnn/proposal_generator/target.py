import mindspore as ms
from mindspore import ops
from ..bbox_utils import bbox2delta


def rpn_anchor_target(anchors,
                      gt_boxes,
                      rpn_batch_size_per_im,
                      rpn_positive_overlap,
                      rpn_negative_overlap,
                      rpn_fg_fraction,
                      use_random=True,
                      batch_size=1,
                      ignore_thresh=-1,
                      is_crowd=None,
                      weights=[1., 1., 1., 1.],
                      assign_on_cpu=False):
    tgt_labels = []
    tgt_bboxes = []
    tgt_deltas = []
    for i in range(batch_size):
        gt_bbox = gt_boxes[i]
        is_crowd_i = is_crowd[i] if is_crowd else None
        # Step1: match anchor and gt_bbox
        '''label_box ok'''
        matches, match_labels = label_box(
            anchors, gt_bbox, rpn_positive_overlap, rpn_negative_overlap, True,
            ignore_thresh, is_crowd_i, assign_on_cpu)
        # Step2: sample anchor
        '''subsample_labels ok but slightly difference with use_random'''
        fg_inds, bg_inds = subsample_labels(match_labels, rpn_batch_size_per_im,
                                            rpn_fg_fraction, 0, use_random=False)
        # Fill with the ignore label (-1), then set positive and negative labels
        labels = ms.numpy.full(match_labels.shape, -1, dtype=ms.int32)
        if bg_inds.shape[0] > 0:
            labels = ops.tensor_scatter_elements(labels, bg_inds, ops.zeros_like(bg_inds))
        if fg_inds.shape[0] > 0:
            labels = ops.tensor_scatter_elements(labels, fg_inds, ops.ones_like(fg_inds))
        # Step3: make output
        if gt_bbox.shape[0] == 0:
            matched_gt_boxes = ops.zeros((matches.shape[0], 4))
            tgt_delta = ops.zeros((matches.shape[0], 4))
        else:
            matched_gt_boxes = gt_bbox.gather(matches, axis=0)
            tgt_delta = bbox2delta(anchors, matched_gt_boxes, weights)
            matched_gt_boxes = ops.stop_gradient(matched_gt_boxes)
            tgt_delta = ops.stop_gradient(tgt_delta)
        labels = ops.stop_gradient(labels)
        tgt_labels.append(labels)
        tgt_bboxes.append(matched_gt_boxes)
        tgt_deltas.append(tgt_delta)

    return tgt_labels, tgt_bboxes, tgt_deltas


def label_box(anchors,
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
        iou = bbox_overlaps(gt_boxes, anchors)
        paddle.set_device(device)

    else:
        iou = bbox_overlaps(gt_boxes, anchors)
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

    iou = iou.transpose()
    matched_vals, matches = iou.top_k(k=1)
    matched_vals = matched_vals.transpose()
    matches = matches.transpose()
    iou = iou.transpose()
    match_labels = ms.numpy.full(matches.shape, -1, dtype=ms.int32)
    # set ignored anchor with iou = -1
    neg_cond = ops.logical_and(matched_vals > -1,
                               matched_vals < negative_overlap)
    match_labels = ms.numpy.where(neg_cond,
                             ops.zeros_like(match_labels), match_labels)
    match_labels = ms.numpy.where(matched_vals >= positive_overlap,
                             ops.ones_like(match_labels), match_labels)
    if allow_low_quality:
        highest_quality_foreach_gt = ops.max(iou, axis=1, keep_dims=True)[1]
        cast = ops.Cast()
        # pred_inds_with_highest_quality = ops.sum(cast(ops.logical_and(iou > 0, iou == highest_quality_foreach_gt),
        #                                               ms.int32), dim=0, keepdim=True)
        pred_inds_with_highest_quality = cast(ops.logical_and(iou > 0, iou == highest_quality_foreach_gt),
                                              ms.int32).sum(axis=0, keepdims=True)
        # pred_inds_with_highest_quality = ops.logical_and(
        #     iou > 0, iou == highest_quality_foreach_gt).cast('int32').sum(
        #     0, keep_dims=True)
        match_labels = ms.numpy.where(pred_inds_with_highest_quality > 0,
                                 ops.ones_like(match_labels),
                                 match_labels)

    matches = matches.flatten()
    match_labels = match_labels.flatten()

    return matches, match_labels


def sample_bbox(matches,
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
        gt_classes = ops.ones(matches.shape, ms.int32) * num_classes
        # return matches, match_labels + num_classes
    else:
        gt_classes = gt_classes.gather(matches, axis=0)
        gt_classes = ms.numpy.where(match_labels == 0,
                               ops.ones_like(gt_classes) * num_classes,
                               gt_classes)
        gt_classes = ms.numpy.where(match_labels == -1,
                                  ops.ones_like(gt_classes) * -1, gt_classes)
    if is_cascade:
        index = ops.arange(matches.shape[0])
        return index, gt_classes
    rois_per_image = int(batch_size_per_im)

    fg_inds, bg_inds = subsample_labels(gt_classes, rois_per_image, fg_fraction,
                                             num_classes, use_random)
    if fg_inds.shape[0] == 0 and bg_inds.shape[0] == 0:
        # fake output labeled with -1 when all boxes are neither
        # foreground nor background
        sampled_inds = ops.zeros([1], dtype='int32')
    else:
        sampled_inds = ops.concat([fg_inds, bg_inds])
    sampled_gt_classes = gt_classes.gather(sampled_inds, axis=0)
    return sampled_inds, sampled_gt_classes


def subsample_labels(labels,
                     num_samples,
                     fg_fraction,
                     bg_label=0,
                     use_random=True):
    positive = ops.nonzero(
        ops.logical_and(labels != -1, labels != bg_label))
    negative = ops.nonzero(labels == bg_label)
    negative = negative.squeeze(-1)
    fg_num = int(num_samples * fg_fraction)
    fg_num = min(positive.numel(), fg_num)
    bg_num = num_samples - fg_num
    bg_num = min(negative.numel(), bg_num)
    if fg_num == 0 and bg_num == 0:
        fg_inds = ops.zeros([0], dtype=ms.int32)
        bg_inds = ops.zeros([0], dtype=ms.int32)
        return fg_inds, bg_inds

    # randomly select positive and negative examples
    cast = ops.Cast()
    negative = cast(negative, ms.int32).flatten()
    # negative = ms.Tensor(negative, dtype=ms.int32).flatten()
    # bg_perm = ops.shuffle(negative)
    # randperm = ops.Randperm(negative.numel(), dtype=ms.int32)
    # bg_perm = randperm(ms.Tensor((negative.numel(),), dtype=ms.int32))
    # TODO: finish slice later
    # bg_perm = paddle.slice(bg_perm, axes=[0], starts=[0], ends=[bg_num])
    # bg_perm = ops.slice(bg_perm, (0,), (bg_num,))
    if use_random:
        # bg_inds = negative.gather(bg_perm, axis=0)
        negative = ops.shuffle(negative)
        # bg_inds = ops.slice(negative, (0,), (bg_num,))
    # else:
        # bg_inds = paddle.slice(negative, axes=[0], starts=[0], ends=[bg_num])
    bg_inds = ops.slice(negative, (0,), (bg_num,))

    if fg_num == 0:
        fg_inds = ops.zeros((0,), ms.int32)
        return fg_inds, bg_inds

    positive = cast(positive, ms.int32).flatten()
    # fg_perm = ops.shuffle(positive)
    # fg_perm = ops.Randperm(positive.numel(), dtype=ms.int32)
    # fg_perm = paddle.slice(fg_perm, axes=[0], starts=[0], ends=[fg_num])
    # fg_perm = ops.slice(fg_perm, (0,), (fg_num,))
    if use_random:
        positive = ops.shuffle(positive)
        # fg_inds = ops.slice(positive, (0,), (fg_num,))
    # else:
        # fg_inds = paddle.slice(positive, axes=[0], starts=[0], ends=[fg_num])
    fg_inds = ops.slice(positive, (0,), (fg_num,))

    return fg_inds, bg_inds


def bbox_overlaps(boxes1, boxes2):
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
    area1 = bbox_area(boxes1)
    area2 = bbox_area(boxes2)

    xy_max = ops.minimum(
        ops.expand_dims(boxes1, 1)[:, :, 2:], boxes2[:, 2:])
    xy_min = ops.maximum(
        ops.expand_dims(boxes1, 1)[:, :, :2], boxes2[:, :2])
    width_height = xy_max - xy_min
    width_height = width_height.clip(xmin=0, xmax=None)
    inter = width_height.prod(axis=2)

    overlaps = ms.numpy.where(inter > 0, inter /
                              (ops.expand_dims(area1, 1) + area2 - inter),
                              ops.zeros_like(inter))
    return overlaps


def bbox_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def generate_proposal_target(rpn_rois,
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
            bbox = ops.concat([rpn_roi, gt_bbox])
        else:
            bbox = rpn_roi

        # Step1: label bbox
        matches, match_labels = label_box(bbox, gt_bbox, fg_thresh, bg_thresh,
                                          False, ignore_thresh, is_crowd_i,
                                          assign_on_cpu)
        # Step2: sample bbox
        '''sample bbox ok if use_random set to False'''
        sampled_inds, sampled_gt_classes = sample_bbox(
            matches, match_labels, gt_class, batch_size_per_im, fg_fraction,
            num_classes, False, is_cascade)

        # Step3: make output
        rois_per_image = bbox if is_cascade else bbox.gather(sampled_inds, axis=0)
        sampled_gt_ind = matches if is_cascade else matches.gather(sampled_inds, axis=0)

        if gt_bbox.shape[0] > 0:
            sampled_bbox = gt_bbox.gather(sampled_gt_ind, axis=0)
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
        new_rois_num.append(ms.Tensor((sampled_inds.shape[0],), dtype=ms.int32))
    new_rois_num = ops.concat(new_rois_num)
    return rois_with_gt, tgt_labels, tgt_bboxes, tgt_gt_inds, new_rois_num
