import mindspore as ms
from mindspore import ops

from bbox_head import BboxHead
from src.FasterRcnn.resnet_from_mindcv import Res5Head
from src.FasterRcnn.proposal_generator.target_layer import BBoxAssigner

def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params


if __name__ == '__main__':
    bbox_assigner = BBoxAssigner
    head = Res5Head(depth=50)
    bbox_head = BboxHead(head=head, bbox_assigner=bbox_assigner, in_channel=2048).set_train(False)
    ms.load_checkpoint('C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.bboxhead.ckpt'
                       , bbox_head)
    # ms_params = mindspore_params(bbox_head)

    body_feats = [ops.ones((1, 1024, 50, 80), dtype=ms.float32)]
    rois = [ops.ones((564, 4), dtype=ms.float32)]
    rois_num = ms.Tensor([564], dtype=ms.int32)
    preds, _ = bbox_head(body_feats, rois, rois_num, None)

    print()
