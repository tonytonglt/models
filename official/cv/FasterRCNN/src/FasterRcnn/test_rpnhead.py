import mindspore as ms
from mindspore import ops

from src.FasterRcnn.proposal_generator.rpn_head import RPNHead
from bbox_head import BboxHead
from src.FasterRcnn.resnet_from_mindcv import Res5Head
from src.FasterRcnn.proposal_generator.target_layer import BBoxAssigner
from src.FasterRcnn.bbox_postprocess import BBoxPostProcess
from src.FasterRcnn.bbox_utils import multiclass_nms
from src.FasterRcnn.bbox_utils import RCNNBox
from src.FasterRcnn.build_fasterrcnn import FasterRCNN


def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape, value)
        ms_params[name] = value
    # return ms_params


if __name__ == '__main__':
    """construct rpn head"""
    rpn_head = RPNHead().set_train(False)
    ms.load_checkpoint('C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.rpnhead.ckpt'
                       , rpn_head)
    # ms_params = mindspore_params(ms_rpn_head)

    """construct bbox head"""
    bbox_assigner = BBoxAssigner()
    head = Res5Head(depth=50)
    bbox_head = BboxHead(head=head, bbox_assigner=bbox_assigner, in_channel=2048, with_pool=True).set_train(False)
    ms.load_checkpoint('C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.bboxhead.ckpt'
                       , bbox_head)
    # mindspore_params(bbox_head)
    """construct bbox postprocessing"""
    rcnnbox = RCNNBox()
    bbox_postprocessing = BBoxPostProcess(decode=rcnnbox, nms=multiclass_nms)


    dummy_body_feats = [ops.ones((1, 1024, 50, 80), type=ms.float32)]
    # import numpy as np
    # dummy_body_feats = [ms.Tensor(np.random.randint(0, 10, size=(1, 1024, 50, 80)), dtype=ms.float32)]
    dummy_inputs = {
        'im_id': ms.Tensor([[0]], dtype=ms.int32),
        'curr_iter': ms.Tensor([0], dtype=ms.int32),
        'image': ops.ones((1, 3, 800, 1267), type=ms.float32),
        'im_shape': ms.Tensor([[800, 1267.32678223]], dtype=ms.float32),
        'scale_factor': ms.Tensor([[1.98019803, 1.98019803]], dtype=ms.float32)
    }

    im_shape = dummy_inputs['im_shape']
    scale_factor = dummy_inputs['scale_factor']
    rois, rois_num, _ = rpn_head(dummy_body_feats, dummy_inputs)
    preds, _ = bbox_head(dummy_body_feats, rois, rois_num, None)
    bbox, bbox_num = bbox_postprocessing(preds, (rois, rois_num), im_shape, scale_factor)

    print()
