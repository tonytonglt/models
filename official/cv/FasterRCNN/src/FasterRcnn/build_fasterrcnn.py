import mindspore as ms
from mindspore import nn, ops

from bbox_head import BboxHead
from src.FasterRcnn.proposal_generator.rpn_head import RPNHead
# from .proposal_generator.rpn_head import RPNHead
from src.FasterRcnn.proposal_generator.target_layer import BBoxAssigner
from src.FasterRcnn.resnet_from_mindcv import Res5Head
from src.FasterRcnn.resnet_from_mindcv import build_resnet50_for_fasterrcnn
from bbox_postprocess import BBoxPostProcess

import mindcv

import paddle



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
        image_tensor = ms.Tensor(self.inputs['image'], dtype=ms.float32)
        image_tensor = ops.transpose(image_tensor, (0, 3, 1, 2))
        body_feats = self.backbone(image_tensor)
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


def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params

def paddle_params(pdparams_file):
    par_dict = paddle.load(pdparams_file)
    pd_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        print(name, parameter.numpy().shape)
        pd_params[name] = parameter.numpy()
    return pd_params
#
# if __name__ == '__main__':
#     pt_params = pytorch_params('C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.pdparams')
#     print()


def build_fasterrcnn_model(training=True):
    # backbone = mindcv.create_model('resnet50', pretrained=False)
    backbone = build_resnet50_for_fasterrcnn()
    rpn_head = RPNHead()
    bbox_assigner = BBoxAssigner()
    head = Res5Head(depth=50)
    bbox_head = BboxHead(head=head, bbox_assigner=bbox_assigner, in_channel=2048, with_pool=True)
    fasterrcnn = FasterRCNN(backbone, rpn_head, bbox_head, training=training)
    return fasterrcnn


if __name__ == '__main__':
    # backbone = mindcv.create_model('resnet50', pretrained=False)
    # rpn_head = RPNHead()
    # bbox_assigner = BBoxAssigner()
    # head = Res5Head(depth=50)
    # bbox_head = BboxHead(head=head, bbox_assigner=bbox_assigner, in_channel=2048)

    # fasterrcnn = FasterRCNN(backbone, rpn_head, bbox_head, None)

    fasterrcnn = build_fasterrcnn_model()
    pd_params = paddle_params('C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.pdparams')
    print('='*40)
    ms_param = mindspore_params(fasterrcnn)
    print()
