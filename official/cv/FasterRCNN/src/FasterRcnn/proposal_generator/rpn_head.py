import mindspore as ms

from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Normal

from .anchor_generator import AnchorGenerator
from .target_layer import RPNTargetAssign
from src.FasterRcnn.proposal_generator.proposal_generator import ProposalGenerator


class RPNFeat(nn.Cell):
    """
    Feature extraction in RPN head

    Args:
        in_channel (int): Input channel
        out_channel (int): Output channel
    """

    def __init__(self, in_channel=1024, out_channel=1024):
        super(RPNFeat, self).__init__()
        # rpn feat is shared with each level
        self.rpn_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=1,
            pad_mode='pad',
            has_bias=True,
            weight_init=Normal(0.01))
        self.relu = nn.ReLU()
        # self.rpn_conv.skip_quant = True  # figure out what is skip_quant? 跳过量化？

    def construct(self, feats):

        rpn_feats = []
        for feat in feats:
            rpn_feats.append(self.relu(self.rpn_conv(feat)))
        return rpn_feats


class RPNHead(nn.Cell):
    """
        Region Proposal Network

        Args:
            anchor_generator (dict): configure of anchor generation
            rpn_target_assign (dict): configure of rpn targets assignment
            train_proposal (dict): configure of proposals generation
                at the stage of training
            test_proposal (dict): configure of proposals generation
                at the stage of prediction
            in_channel (int): channel of input feature maps which can be
                derived by from_config
    """

    def __init__(self,
                 # anchor_generator=_get_class_default_kwargs(AnchorGenerator),
                 # rpn_target_assign=_get_class_default_kwargs(RPNTargetAssign),
                 # train_proposal=_get_class_default_kwargs(ProposalGenerator,
                 #                                          12000, 2000),
                 # test_proposal=_get_class_default_kwargs(ProposalGenerator),
                 anchor_generator={'aspect_ratios': [0.5, 1.0, 2.0], 'anchor_sizes': [32, 64, 128, 256, 512], 'strides': [16]},
                 rpn_target_assign={'batch_size_per_im': 256, 'fg_fraction': 0.5, 'negative_overlap': 0.3, 'positive_overlap': 0.7, 'use_random': True},
                 train_proposal={'min_size': 0.0, 'nms_thresh': 0.7, 'pre_nms_top_n': 12000, 'post_nms_top_n': 2000, 'topk_after_collect': False},
                 test_proposal={'min_size': 0.0, 'nms_thresh': 0.7, 'pre_nms_top_n': 6000, 'post_nms_top_n': 1000},
                 in_channel=1024,
                 loss_rpn_bbox=None):
        super(RPNHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)
        if isinstance(train_proposal, dict):
            self.train_proposal = ProposalGenerator(**train_proposal)
        if isinstance(test_proposal, dict):
            self.test_proposal = ProposalGenerator(**test_proposal)
        self.loss_rpn_bbox = loss_rpn_bbox

        num_anchors = self.anchor_generator.num_anchors
        self.rpn_feat = RPNFeat(in_channel, in_channel)

        self.rpn_rois_score = nn.Conv2d(
            in_channels=in_channel,
            out_channels=num_anchors,
            kernel_size=1,
            padding=0,
            has_bias=True,
            weight_init=Normal(0.01))

        self.rpn_rois_delta = nn.Conv2d(
            in_channels=in_channel,
            out_channels=4 * num_anchors,
            kernel_size=1,
            padding=0,
            has_bias=True,
            weight_init=Normal(0.01))

    @classmethod
    def from_config(cls, cfg, input_shape):
        # FPN share same rpn head
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels}

    def construct(self, feats, inputs):
        '''rpn_feats ok'''
        feats = [feats]
        rpn_feats = self.rpn_feat(feats)
        scores = []
        deltas = []

        for rpn_feat in rpn_feats:
            '''scores/deltas ok'''
            rrs = self.rpn_rois_score(rpn_feat)
            rrd = self.rpn_rois_delta(rpn_feat)
            scores.append(rrs)
            deltas.append(rrd)

        '''anchors ok'''
        anchors = self.anchor_generator(rpn_feats)

        '''rois ok: rois 443 & 443'''
        rois, rois_num = self._gen_proposal(scores, deltas, anchors, inputs)
        if self.training:
            '''loss_rpn_cls: 有些许误差(use random), loss_rpn_reg完全相同'''
            loss = self.get_loss(scores, deltas, anchors, inputs)
            return rois, rois_num, loss
        else:
            return rois, rois_num, None

    def _gen_proposal(self, scores, bbox_deltas, anchors, inputs):
        """
        scores (list[Tensor]): Multi-level scores prediction
        bbox_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info
        """
        prop_gen = self.train_proposal if self.training else self.test_proposal
        # im_shape = inputs['im_shape']
        # im_shape = ms.Tensor([[inputs['h'].item(), inputs['w'].item()]], dtype=ms.float32)
        im_shape = ms.Tensor(inputs['image'].shape[-2:], ms.float32).reshape(-1, 2)

        # Collect multi-level proposals for each batch
        # Get 'topk' of them as final output

        bs_rois_collect = []
        bs_rois_num_collect = []

        # batch_size = paddle.slice(paddle.shape(im_shape), [0], [0], [1])
        batch_size = ops.slice(ms.Tensor(im_shape.shape), (0,), (1,))

        # Generate proposals for each level and each batch.
        # Discard batch-computing to avoid sorting bbox cross different batches.
        for i in range(batch_size):
            rpn_rois_list = []
            rpn_prob_list = []
            rpn_rois_num_list = []

            for rpn_score, rpn_delta, anchor in zip(scores, bbox_deltas,
                                                    anchors):
                rpn_rois, rpn_rois_prob, rpn_rois_num, post_nms_top_n = prop_gen(  # TODO: attention!!!!
                    scores=rpn_score[i:i + 1],
                    bbox_deltas=rpn_delta[i:i + 1],
                    anchors=anchor,
                    im_shape=im_shape[i:i + 1])

                rpn_rois = ops.stop_gradient(rpn_rois)
                rpn_rois_prob = ops.stop_gradient(rpn_rois_prob)

                rpn_rois_list.append(rpn_rois)
                rpn_prob_list.append(rpn_rois_prob)
                rpn_rois_num_list.append(rpn_rois_num)

            if len(scores) > 1:
                rpn_rois = ops.concat(rpn_rois_list)
                rpn_prob = ops.concat(rpn_prob_list).flatten()

                num_rois = ops.cast(rpn_prob.shape[0], ms.int32)
                if num_rois > post_nms_top_n:
                    topk_prob, topk_inds = rpn_prob.topk(post_nms_top_n)
                    topk_rois = rpn_rois.gather(topk_inds)
                else:
                    topk_rois = rpn_rois
                    topk_prob = rpn_prob
            else:
                topk_rois = rpn_rois_list[0]
                topk_prob = rpn_prob_list[0].flatten()

            bs_rois_collect.append(topk_rois)
            bs_rois_num_collect.append(ms.Tensor((topk_rois.shape[0], ), dtype=ms.int32))

        bs_rois_num_collect = ops.concat(bs_rois_num_collect)

        output_rois = bs_rois_collect
        output_rois_num = bs_rois_num_collect

        return output_rois, output_rois_num

    def get_loss(self, pred_scores, pred_deltas, anchors, inputs):
        """
        pred_scores (list[Tensor]): Multi-level scores prediction
        pred_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info, including im, gt_bbox, gt_score
        """
        anchors = [a.reshape((-1, 4)) for a in anchors]
        anchors = ops.concat(anchors)

        scores = [
            ops.reshape(
                ops.transpose(
                    v, input_perm=(0, 2, 3, 1)),
                input_shape=(v.shape[0], -1, 1)) for v in pred_scores
        ]
        scores = ops.concat(scores, axis=1)

        deltas = [
            ops.reshape(
                ops.transpose(
                    v, input_perm=(0, 2, 3, 1)),
                input_shape=(v.shape[0], -1, 4)) for v in pred_deltas
        ]
        deltas = ops.concat(deltas, axis=1)

        '''rpn_target_assign ok w/o random'''
        score_tgt, bbox_tgt, loc_tgt, norm = self.rpn_target_assign(inputs,
                                                                    anchors)

        scores = scores.reshape(-1, )
        deltas = deltas.reshape(-1, 4)

        score_tgt = ops.concat(score_tgt)
        score_tgt = ops.stop_gradient(score_tgt)

        pos_mask = score_tgt == 1
        pos_ind = pos_mask.nonzero()

        valid_mask = score_tgt >= 0
        valid_ind = valid_mask.nonzero()

        # cls loss
        if valid_ind.shape[0] == 0:
            loss_rpn_cls = ops.zeros((1, ), dtype=ms.float32)
        else:
            score_pred = scores.gather(valid_ind, axis=0)
            score_label = ops.cast(score_tgt.gather(valid_ind, axis=0), ms.float32)
            score_label = ops.stop_gradient(score_label)
            weight = ops.ones(score_pred.shape, ms.float32)
            pos_weight = ops.ones(score_pred.shape, ms.float32)
            loss_rpn_cls = ops.binary_cross_entropy_with_logits(
                logits=score_pred, label=score_label, weight=weight, pos_weight=pos_weight, reduction="sum")

        # reg loss
        if pos_ind.shape[0] == 0:
            loss_rpn_reg = ops.zeros((1, ), ms.float32)
        else:
            loc_pred = deltas.gather(pos_ind, axis=0)
            loc_tgt = ops.concat(loc_tgt)
            loc_tgt = loc_tgt.gather(pos_ind, axis=0)
            loc_tgt = ops.stop_gradient(loc_tgt)

            if self.loss_rpn_bbox is None:
                loss_rpn_reg = ops.abs(loc_pred - loc_tgt).sum()
            else:
                loss_rpn_reg = self.loss_rpn_bbox(loc_pred, loc_tgt).sum()

        return {
            'loss_rpn_cls': loss_rpn_cls / norm,
            'loss_rpn_reg': loss_rpn_reg / norm
        }
