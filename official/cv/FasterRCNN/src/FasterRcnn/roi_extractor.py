import mindspore as ms

# from mindspore.ops import ROIAlign, cat
# from mindspore.ops.operations import ROIAlign # ???
import mindspore as ms
from mindspore import ops


def _to_list(v):
    if not isinstance(v, (list, tuple)):
        return [v]
    return v


class RoIAlign(object):
    def __init__(self,
                 resolution=14,
                 spatial_scale=0.0625,
                 sampling_ratio=0,
                 canconical_level=4,
                 canonical_size=224,
                 start_level=0,
                 end_level=3,
                 aligned=False):
        super(RoIAlign, self).__init__()
        self.resolution = resolution
        self.spatial_scale = _to_list(spatial_scale)
        self.sampling_ratio = sampling_ratio
        self.canconical_level = canconical_level
        self.canonical_size = canonical_size
        self.start_level = start_level
        self.end_level = end_level
        self.aligned = aligned

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'spatial_scale': [1. / i.stride for i in input_shape]}

    def __call__(self, feats, roi, rois_num):
        roi = ops.concat(roi) if len(roi) > 1 else roi[0]
        # roi = roi[:1]
        if len(feats) == 1:  # w/o. FPN
            roi_align = ops.ROIAlign(pooled_height=self.resolution,
                                     pooled_width=self.resolution,
                                     spatial_scale=self.spatial_scale[0],
                                     sample_num=2,
                                     roi_end_mode=0)  # params: (pooled_height, pooled_width, spatial_scale, sample_num=2, roi_end_mode=1)
            # TODO: rois shape convert to [rois_n,5], rois_n 为RoI的数量。第二个维度的大小必须为 5 ，分别代表 (image_index,top_left_x,top_left_y,bottom_right_x,bottom_right_y)
            idx = ops.zeros((roi.shape[0], 1), ms.float32)
            rois = ops.concat((idx, roi), axis=1)
            rois_feat = roi_align(feats[0], rois)  # params(features, rois) features (Tensor) - 输入特征，shape: (N,C,H,W), rois - shape: (rois_n,5)
            # roi_align paddle实现
            # rois_feat = paddle.vision.ops.roi_align(
            #     x=feats[self.start_level],
            #     boxes=roi,
            #     boxes_num=rois_num,
            #     output_size=self.resolution,
            #     spatial_scale=self.spatial_scale[0],
            #     aligned=self.aligned)

        # else:  # with FPN
        #     offset = 2
        #     k_min = self.start_level + offset
        #     k_max = self.end_level + offset
        #     rois_feat = None
        #     """
        #     paddle: distribute_fpn_proposals
        #     In Feature Pyramid Networks (FPN) models, it is needed to distribute
        #     all proposals into different FPN level, with respect to scale of the proposals,
        #     the referring scale and the referring level.
        #     """
        #     rois_dist, restore_index, rois_num_dist = distribute_fpn_proposals(
        #         roi,
        #         k_min,
        #         k_max,
        #         self.canconical_level,
        #         self.canonical_size,
        #         rois_num=rois_num)

        return rois_feat
