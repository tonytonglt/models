import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Normal
from roi_extractor import RoIAlign
from bbox_utils import bbox2delta
from resnet_from_mindcv import Res5Head


class BboxHead(nn.Cell):
    def __init__(self,
                 head,
                 in_channel,
                 # roi_extractor=_get_class_default_kwargs(RoIAlign),
                 roi_extractor={'resolution': 14, 'sampling_ratio': 0, 'aligned': True, 'spatial_scale': [0.0625]},
                 bbox_assigner='BboxAssigner',
                 with_pool=False,
                 num_classes=80,
                 bbox_weight=[10., 10., 5., 5.],
                 bbox_loss=None,
                 loss_normalize_pos=False):
        super(BboxHead, self).__init__()
        self.head = head
        # TODO: finish roi extractor
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.bbox_assigner = bbox_assigner  # TODO: construct bbox_assigner object, <ppdet.modeling.proposal_generator.target_layer.BBoxAssigner object at 0x00000213C8736748>
        self.with_pool = with_pool
        self.num_classes = num_classes
        self.bbox_weight = bbox_weight
        self.bbox_loss = bbox_loss
        self.loss_normalize_pos = loss_normalize_pos

        self.bbox_score = nn.Dense(in_channel, self.num_classes + 1, weight_init=Normal(0.01))  # weight_init
        # self.bbox_delta.skip_quant = True #  ???
        self.bbox_delta = nn.Dense(in_channel, 4 * self.num_classes, weight_init=Normal(0.001))

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        # head = create(cfg['head'], **kwargs) # TODO: construct the head network(e.g. Res5Head)
        head = Res5Head(depth=50)
        return {
            'roi_extractor': roi_pooler,
            'head': head,
            'in_channel': head.out_shape[0].channels
        }

    def construct(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        if self.training:
            rois, rois_num, targets = self.bbox_assigner(rois, rois_num, inputs)  # TODO: construct bbox_assigner object as in line 27
            self.assigned_rois = (rois, rois_num)
            self.assigned_targets = targets

        body_feats = [body_feats]

        rois_feat = self.roi_extractor(body_feats, rois, rois_num)  # body feats need to be list

        rois_feat = ops.ones((512, 1024, 14, 14), ms.float32)

        bbox_feat = self.head(rois_feat)


        if self.with_pool:
            pool = nn.AdaptiveAvgPool2d(output_size=1)  # only supported on GPU, use avgpool for testing
            # pool = nn.AvgPool2d(kernel_size=7, stride=1)
            feat = pool(bbox_feat)
            feat = feat.squeeze(axis=(2, 3))
        else:
            feat = bbox_feat
        scores = self.bbox_score(feat)
        deltas = self.bbox_delta(feat)

        if self.training:
            loss = self.get_loss(
                scores,
                deltas,
                targets,
                rois,
                self.bbox_weight,
                loss_normalize_pos=self.loss_normalize_pos)
            return loss, bbox_feat
        else:
            pred = self.get_prediction(scores, deltas)
            return pred, self.head


    def get_loss(self,
                 scores,
                 deltas,
                 targets,
                 rois,
                 bbox_weight,
                 loss_normalize_pos=False):
        """
        scores (Tensor): scores from bbox head outputs
        deltas (Tensor): deltas from bbox head outputs
        targets (list[List[Tensor]]): bbox targets containing tgt_labels, tgt_bboxes and tgt_gt_inds
        rois (List[Tensor]): RoIs generated in each batch
        """
        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        loss_bbox = {}

        # TODO: better pass args
        tgt_labels, tgt_bboxes, tgt_gt_inds = targets

        # bbox cls
        tgt_labels = ops.concat(tgt_labels) if len(
            tgt_labels) > 1 else tgt_labels[0]
        valid_inds = ops.nonzero(tgt_labels >= 0).flatten()
        if valid_inds.shape[0] == 0:
            loss_bbox[cls_name] = ops.zeros(1, dtype=ms.float32)
        else:
            tgt_labels = ops.cast(tgt_labels, ms.int32)  # target 数据类型仅支持int32
            tgt_labels = ops.stop_gradient(tgt_labels)

            if not loss_normalize_pos:
                loss_bbox_cls = ops.cross_entropy(
                    inputs=scores, target=tgt_labels, reduction='mean')  # target 数据类型仅支持int32
            else:
                loss_bbox_cls = ops.cross_entropy(
                    inputs=scores, target=tgt_labels,
                    reduction='none').sum() / (tgt_labels.shape[0] + 1e-7)
            loss_bbox[cls_name] = loss_bbox_cls

        # bbox reg loss

        cls_agnostic_bbox_reg = deltas.shape[1] == 4

        fg_inds = ops.nonzero(
            ops.logical_and(tgt_labels >= 0, tgt_labels <
                            self.num_classes))

        if fg_inds.numel() == 0:
            loss_bbox[reg_name] = ops.zeros(1, ms.float32)
            return loss_bbox

        fg_inds = fg_inds.flatten()

        if cls_agnostic_bbox_reg:
            reg_delta = deltas.gather(fg_inds, axis=0)
        else:
            fg_gt_classes = tgt_labels.gather(fg_inds, axis=0)

            reg_row_inds = ops.arange(fg_gt_classes.shape[0]).unsqueeze(1)
            reg_row_inds = reg_row_inds.tile((1, 4)).reshape((-1, 1))

            reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + ops.arange(4)

            reg_col_inds = reg_col_inds.reshape((-1, 1))
            reg_inds = ops.concat([reg_row_inds, reg_col_inds], axis=1)

            reg_delta = deltas.gather(fg_inds, axis=0)
            reg_delta = reg_delta.gather_nd(reg_inds).reshape((-1, 4))
        rois = ops.concat(rois) if len(rois) > 1 else rois[0]
        tgt_bboxes = ops.concat(tgt_bboxes) if len(tgt_bboxes) > 1 else tgt_bboxes[0]

        reg_target = bbox2delta(rois, tgt_bboxes, bbox_weight)
        reg_target = reg_target.gather(fg_inds, axis=0)
        reg_target = ops.stop_gradient(reg_target)

        if self.bbox_loss is not None:
            reg_delta = self.bbox_transform(reg_delta)
            reg_target = self.bbox_transform(reg_target)

            if not loss_normalize_pos:
                loss_bbox_reg = self.bbox_loss(
                    reg_delta, reg_target).sum() / tgt_labels.shape[0]
                loss_bbox_reg *= self.num_classes

            else:
                loss_bbox_reg = self.bbox_loss(
                    reg_delta, reg_target).sum() / (tgt_labels.shape[0] + 1e-7)

        else:
            loss_bbox_reg = (reg_delta - reg_target).abs().sum() / tgt_labels.shape[0]

        loss_bbox[reg_name] = loss_bbox_reg

        return loss_bbox

    def bbox_transform(self, deltas, weights=[0.1, 0.1, 0.2, 0.2]):
        wx, wy, ww, wh = weights

        deltas = deltas.reshape((0, -1, 4))

        # TODO: finish slice later
        # dx = paddle.slice(deltas, axes=[2], starts=[0], ends=[1]) * wx
        # dy = paddle.slice(deltas, axes=[2], starts=[1], ends=[2]) * wy
        # dw = paddle.slice(deltas, axes=[2], starts=[2], ends=[3]) * ww
        # dh = paddle.slice(deltas, axes=[2], starts=[3], ends=[4]) * wh

        dx = ops.slice(deltas, (0, 0), (len(deltas), 1)) * wx
        dy = ops.slice(deltas, (0, 1), (len(deltas), 1)) * wy
        dw = ops.slice(deltas, (0, 2), (len(deltas), 1)) * ww
        dh = ops.slice(deltas, (0, 3), (len(deltas), 1)) * wh

        dw = dw.clip(-1.e10, np.log(1000. / 16))
        dh = dh.clip(-1.e10, np.log(1000. / 16))

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = dw.exp()
        pred_h = dh.exp()

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        x1 = x1.reshape((-1,))
        y1 = y1.reshape((-1,))
        x2 = x2.reshape((-1,))
        y2 = y2.reshape((-1,))

        return ops.concat([x1, y1, x2, y2])

    def get_prediction(self, score, delta):
        bbox_prob = ops.softmax(score)
        return delta, bbox_prob

    def get_head(self, ):
        return self.head

    def get_assigned_targets(self, ):
        return self.assigned_targets

    def get_assigned_rois(self, ):
        return self.assigned_rois


# if __name__ == '__main__':
#     head = 0
#     in_channel = 64
#     bboxhead = BboxHead(head, in_channel)
#     print()



    """
    mindspore.nn.Dense(
        in_channels,
        out_channels,
        weight_init='normal',
        bias_init='zeros',
        has_bias=True,
        activation=None
    )(x) -> Tensor
    """



