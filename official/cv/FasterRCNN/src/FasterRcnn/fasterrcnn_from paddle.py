from mindspore import nn, ops

class FasterRCNN(nn.Cell):
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 bbox_post_process,
                 neck=None,
                 training=False
                 ):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process
        self.inputs = {}
        self.training = training

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        bbox_head = create(cfg['bbox_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "bbox_head": bbox_head,
        }

    def construct(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)
            bbox_loss, _ = self.bbox_head(body_feats, rois, rois_num,
                                          self.inputs)
            return rpn_loss, bbox_loss
        else:
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            preds, _ = self.bbox_head(body_feats, rois, rois_num, None)

            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            bbox, bbox_num = self.bbox_post_process(preds, (rois, rois_num),
                                                    im_shape, scale_factor)

            # rescale the prediction back to origin image
            bboxes, bbox_pred, bbox_num = self.bbox_post_process.get_pred(
                bbox, bbox_num, im_shape, scale_factor)
            return bbox_pred, bbox_num

    def get_loss(self):
        rpn_loss, bbox_loss = self.construct()
        loss = {}
        loss.update(rpn_loss)
        loss.update(bbox_loss)
        total_loss = ops.addn(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num = self.construct()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output
