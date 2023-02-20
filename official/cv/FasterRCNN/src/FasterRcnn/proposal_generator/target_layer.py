from .target import generate_proposal_target, rpn_anchor_target
import mindspore as ms

class RPNTargetAssign(object):
    """
    RPN targets assignment module

    The assignment consists of three steps:
        1. Match anchor and ground-truth box, label the anchor with foreground
           or background sample
        2. Sample anchors to keep the properly ratio between foreground and 
           background
        3. Generate the targets for classification and regression branch


    Args:
        batch_size_per_im (int): Total number of RPN samples per image. 
            default 256
        fg_fraction (float): Fraction of anchors that is labeled
            foreground, default 0.5
        positive_overlap (float): Minimum overlap required between an anchor
            and ground-truth box for the (anchor, gt box) pair to be 
            a foreground sample. default 0.7
        negative_overlap (float): Maximum overlap allowed between an anchor
            and ground-truth box for the (anchor, gt box) pair to be 
            a background sample. default 0.3
        ignore_thresh(float): Threshold for ignoring the is_crowd ground-truth
            if the value is larger than zero.
        use_random (bool): Use random sampling to choose foreground and 
            background boxes, default true.
        assign_on_cpu (bool): In case the number of gt box is too large, 
            compute IoU on CPU, default false.
    """

    def __init__(self,
                 batch_size_per_im=256,
                 fg_fraction=0.5,
                 positive_overlap=0.7,
                 negative_overlap=0.3,
                 ignore_thresh=-1.,
                 use_random=True,
                 assign_on_cpu=False):
        super(RPNTargetAssign, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.ignore_thresh = ignore_thresh
        self.use_random = use_random
        self.assign_on_cpu = assign_on_cpu

    def __call__(self, inputs, anchors):
        """
        inputs: ground-truth instances.
        anchor_box (Tensor): [num_anchors, 4], num_anchors are all anchors in all feature maps.
        """
        gt_boxes = ms.Tensor(inputs['gt_bbox'], dtype=ms.float32)
        is_crowd = inputs.get('is_crowd', None)
        batch_size = len(gt_boxes)
        tgt_labels, tgt_bboxes, tgt_deltas = rpn_anchor_target(
            anchors,
            gt_boxes,
            self.batch_size_per_im,
            self.positive_overlap,
            self.negative_overlap,
            self.fg_fraction,
            self.use_random,
            batch_size,
            self.ignore_thresh,
            is_crowd,
            assign_on_cpu=self.assign_on_cpu)
        norm = self.batch_size_per_im * batch_size

        return tgt_labels, tgt_bboxes, tgt_deltas, norm


class BBoxAssigner(object):
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
        gt_classes = ms.Tensor(inputs['gt_class'], dtype=ms.float32)
        gt_boxes = ms.Tensor(inputs['gt_bbox'], dtype=ms.float32)
        is_crowd = inputs.get('is_crowd', None)
        # rois, tgt_labels, tgt_bboxes, tgt_gt_inds
        # new_rois_num
        outs = generate_proposal_target(
            rpn_rois, gt_classes, gt_boxes, self.batch_size_per_im,
            self.fg_fraction, self.fg_thresh, self.bg_thresh, self.num_classes,
            self.ignore_thresh, is_crowd, self.use_random, is_cascade,
            self.cascade_iou[stage], self.assign_on_cpu, add_gt_as_proposals)
        rois = outs[0]
        rois_num = outs[-1]
        # tgt_labels, tgt_bboxes, tgt_gt_inds
        targets = outs[1:4]
        return rois, rois_num, targets