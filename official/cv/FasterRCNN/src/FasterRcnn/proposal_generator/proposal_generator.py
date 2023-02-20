import mindspore as ms
from mindspore import ops
from ..bbox_utils import delta2bbox
from ..boxes import Boxes


class ProposalGenerator(object):
    """
    Proposal generation module

    For more details, please refer to the document of generate_proposals
    in ppdet/modeing/ops.py

    Args:
        pre_nms_top_n (int): Number of total bboxes to be kept per
            image before NMS. default 6000
        post_nms_top_n (int): Number of total bboxes to be kept per
            image after NMS. default 1000
        nms_thresh (float): Threshold in NMS. default 0.5
        min_size (flaot): Remove predicted boxes with either height or
             width < min_size. default 0.1
        eta (float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
             `adaptive_threshold = adaptive_threshold * eta` in each iteration.
             default 1.
        topk_after_collect (bool): whether to adopt topk after batch
             collection. If topk_after_collect is true, box filter will not be
             used after NMS at each image in proposal generation. default false
    """

    def __init__(self,
                 pre_nms_top_n=12000,
                 post_nms_top_n=2000,
                 nms_thresh=.5,
                 min_size=.1,
                 eta=1.,
                 topk_after_collect=False):
        super(ProposalGenerator, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta
        self.topk_after_collect = topk_after_collect

    def __call__(self, scores, bbox_deltas, anchors, im_shape):

        top_n = self.pre_nms_top_n if self.topk_after_collect else self.post_nms_top_n

        """re-implement of detectron2"""
        N = scores.shape[0]
        B = anchors.shape[-1]
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.unsqueeze(dim=0).transpose(0, 2, 3, 1).reshape(N, -1)
            for score in scores
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            delta.reshape(1, -1, anchors.shape[-1], delta.shape[-2], delta.shape[-1])
            .transpose([0, 3, 4, 1, 2])
            .reshape(N, -1, B)
            for delta in bbox_deltas
        ]

        anchors = anchors.unsqueeze(0)
        rpn_rois, rpn_rois_prob = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, im_shape
        )
        rpn_rois_num = len(rpn_rois)

        """paddle implementation"""
        # variances = ops.ones_like(anchors)
        # if hasattr(paddle.vision.ops, "generate_proposals"):
        #     rpn_rois, rpn_rois_prob, rpn_rois_num = paddle.vision.ops.generate_proposals(
        #         scores,
        #         bbox_deltas,
        #         im_shape,
        #         anchors,
        #         variances,
        #         pre_nms_top_n=self.pre_nms_top_n,
        #         post_nms_top_n=top_n,
        #         nms_thresh=self.nms_thresh,
        #         min_size=self.min_size,
        #         eta=self.eta,
        #         return_rois_num=True)
        # else:
        #     rpn_rois, rpn_rois_prob, rpn_rois_num = ops.generate_proposals(
        #         scores,
        #         bbox_deltas,
        #         im_shape,
        #         anchors,
        #         variances,
        #         pre_nms_top_n=self.pre_nms_top_n,
        #         post_nms_top_n=top_n,
        #         nms_thresh=self.nms_thresh,
        #         min_size=self.min_size,
        #         eta=self.eta,
        #         return_rois_num=True)

        return rpn_rois, rpn_rois_prob, rpn_rois_num, self.post_nms_top_n

    def predict_proposals(self,
                          anchors,
                          pred_objectness_logits,
                          pred_anchor_deltas,
                          image_sizes):
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return self.find_top_rpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self.nms_thresh,
            self.pre_nms_top_n,
            self.post_nms_top_n,
            self.min_size,
            # self.training,
        )

    def _decode_proposals(self, anchors, pred_anchor_deltas):
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.shape[-1]
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.unsqueeze(0).expand(ms.Tensor((N, -1, -1), dtype=ms.int32)).reshape([-1, B])
            proposals_i = delta2bbox(pred_anchor_deltas_i, anchors_i, [10., 10., 5., 5.])  # [10., 10., 5., 5.]
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.reshape(N, -1, B))
        return proposals

    def find_top_rpn_proposals(self,
                               proposals,
                               pred_objectness_logits,
                               image_sizes,
                               nms_thresh,
                               pre_nms_topk,
                               post_nms_topk,
                               min_box_size,
                               # training,
                               ):
        num_images = len(image_sizes)  # TODO: not right?
        # 1. Select top-k anchor for every level and every image
        topk_scores = []  # #lvl Tensor, each of shape N x topk
        topk_proposals = []
        level_ids = []  # #lvl Tensor, each of shape (topk,)
        batch_idx = ops.arange(num_images)

        for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
            Hi_Wi_A = logits_i.shape[1]
            if isinstance(Hi_Wi_A, ms.Tensor):  # it's a tensor in tracing
                num_proposals_i = ops.clip(Hi_Wi_A, max=pre_nms_topk)
            else:
                num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

            topk_scores_i, topk_idx = ops.topk(logits_i, num_proposals_i, dim=1)

            # each is N x topk
            # topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4
            topk_proposals_i = ops.gather(proposals_i.squeeze(0), topk_idx[0], axis=0).unsqueeze(0)  # N x topk x 4
            topk_proposals.append(topk_proposals_i)
            topk_scores.append(topk_scores_i)
            level_ids.append(ops.full((num_proposals_i,), level_id))

        # 2. Concat all levels together
        topk_scores = ops.concat(topk_scores, axis=1)
        topk_proposals = ops.concat(topk_proposals, axis=1)
        level_ids = ops.concat(level_ids, axis=0)

        # 3. For each image, run a per-level NMS, and choose topk results.
        result = []
        for n, image_size in enumerate(image_sizes):
            boxes = Boxes(topk_proposals[n])
            scores_per_img = topk_scores[n]
            lvl = level_ids

            valid_mask = ops.isfinite(boxes.tensor).all(axis=1) & ops.isfinite(scores_per_img)  # TODO: attention!!!!
            if not valid_mask.all():
                boxes = boxes[valid_mask]
                scores_per_img = scores_per_img[valid_mask]
                lvl = lvl[valid_mask]
            boxes.clip(image_size)

            # filter empty boxes
            keep = boxes.nonempty(threshold=min_box_size)  # TODO: attention!!!! 可能会有全部为0的情况
            if keep.sum() != len(boxes):
                boxes = ms.Tensor(boxes.tensor.asnumpy()[keep.asnumpy()], dtype=ms.float32)
                scores_per_img = ms.Tensor(scores_per_img.asnumpy()[keep.asnumpy()], dtype=ms.float32)
                lvl = ms.Tensor(lvl.asnumpy()[keep.asnumpy()], dtype=ms.float32)
                # boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

            # keep_dets = paddle.concat([scores_per_img.unsqueeze(1), boxes.tensor], axis=1)
            # keep_dets = nms(keep_dets, match_threshold=nms_thresh)

            # keep_dets_idx = paddle.vision.ops.nms(boxes.tensor, iou_threshold=nms_thresh, scores=scores_per_img)
            box_with_scores = ops.concat((boxes.tensor, scores_per_img.unsqueeze(1)), axis=1)
            nms = ops.NMSWithMask(iou_threshold=nms_thresh)
            ordered_box, ordered_idx, valid_mask = nms(box_with_scores)
            keep_idx = ms.Tensor(ordered_idx.asnumpy()[valid_mask.asnumpy()], dtype=ms.int32)

            keep_dets_boxes = ops.gather(boxes.tensor, keep_idx, axis=0)  # boxes are sorted by scores
            keep_dets_scores = ops.gather(scores_per_img, keep_idx, axis=0)

            # TODO: release next line batched_nms
            # keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)

            # In Detectron1, there was different behavior during training vs. testing.
            # (https://github.com/facebookresearch/Detectron/issues/459)
            # During training, topk is over the proposals from *all* images in the training batch.
            # During testing, it is over the proposals for each image separately.
            # As a result, the training behavior becomes batch-dependent,
            # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
            # This bug is addressed in Detectron2 to make the behavior independent of batch size.

            keep_dets_boxes = keep_dets_boxes[:post_nms_topk]  # keep is already sorted
            keep_dets_scores = keep_dets_scores[:post_nms_topk]
            # result = keep
            return keep_dets_boxes, keep_dets_scores