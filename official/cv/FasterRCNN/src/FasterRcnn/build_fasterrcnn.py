import mindspore as ms
from mindspore import nn, ops

from bbox_head import BboxHead
from src.FasterRcnn.proposal_generator.rpn_head import RPNHead
# from .proposal_generator.rpn_head import RPNHead
from src.FasterRcnn.proposal_generator.target_layer import BBoxAssigner
from src.FasterRcnn.resnet_from_mindcv import Res5Head
from src.FasterRcnn.resnet_from_mindcv import build_resnet50_for_fasterrcnn
from bbox_postprocess import BBoxPostProcess

# import paddle
import numpy as np



class FasterRCNN(nn.Cell):
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 bbox_post_process=None,
                 neck=None,
                 training=False
                 ):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process
        self.training = training
        self.inputs = {}

    def construct(self, inputs):
        self.inputs = inputs
        # image_tensor = ms.Tensor(self.inputs['image'], dtype=ms.float32)
        # image_tensor = ops.transpose(image_tensor, (0, 3, 1, 2))
        # image_tensor = self.inputs['image']
        body_feats = self.backbone(self.inputs['image'])
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        if self.training:
            # body_feats = ms.Tensor(np.load('C:\\tongli\\input_1\\body_feats.npz')['arr_0'], ms.float32)
            # self.inputs['image'] = np.load('C:\\tongli\\input_1\\inputs_image.npz')['arr_0']
            # self.inputs['gt_bbox'] = ms.Tensor(np.load('C:\\tongli\\input_1\\inputs_gt_bbox.npz')['arr_0'], ms.float32).reshape(1, 1, 4)
            # self.inputs['gt_class'] = ms.Tensor(np.load('C:\\tongli\\input_1\\inputs_gt_class.npz')['arr_0'], ms.float32).reshape(1, 1, 1)
            # self.inputs['h'] = np.array(1078, np.float32)
            # self.inputs['w'] = np.array(800, np.float32)

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


def build_fasterrcnn_model(training=True):
    # backbone = mindcv.create_model('resnet50', pretrained=False)
    backbone = build_resnet50_for_fasterrcnn()
    ms.load_checkpoint('C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.backbone.ckpt'
                       , backbone)
    rpn_head = RPNHead()
    ms.load_checkpoint('C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.rpnhead.ckpt'
                       , rpn_head)
    bbox_assigner = BBoxAssigner()
    head = Res5Head(depth=50)
    bbox_head = BboxHead(head=head, bbox_assigner=bbox_assigner, in_channel=2048, with_pool=True)
    ms.load_checkpoint('C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.bboxhead.ckpt'
                       , bbox_head)
    fasterrcnn = FasterRCNN(backbone, rpn_head, bbox_head, training=training)
    return fasterrcnn

