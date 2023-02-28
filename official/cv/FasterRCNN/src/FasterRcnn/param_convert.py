import mindspore as ms
import paddle

import mindcv

from bbox_head import BboxHead
from src.FasterRcnn.proposal_generator.rpn_head import RPNHead
from src.FasterRcnn.proposal_generator.target_layer import BBoxAssigner
from src.FasterRcnn.resnet_from_mindcv import Res5Head
from src.FasterRcnn.build_fasterrcnn import FasterRCNN
from src.FasterRcnn.bbox_head_params_dic import bbox_head_param_dic
from src.FasterRcnn.backbone_params_dic import backbone_param_dic
from src.FasterRcnn.resnet_from_mindcv import build_resnet50_for_fasterrcnn
from src.FasterRcnn.resnet_from_mindcv import ResNet50ForFasterrcnn
from src.FasterRcnn.build_fasterrcnn import build_fasterrcnn_model


# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def paddle_params(pdparams_file):
    par_dict = paddle.load(pdparams_file)
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        print(name)
        # print(name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params


# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        # print(name, value.shape)
        print(name)
        ms_params[name] = value
    return ms_params


def param_convert(ms_params, pt_params, ckpt_path):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}
    new_params_list = []
    for ms_param in ms_params.keys():
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数
        if "bn" in ms_param or "downsample.1" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表
            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    new_ckpt_path = 'C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.msckpt'
    ms.save_checkpoint(new_params_list, new_ckpt_path)


def rpnhead_param_convert(ms_params, pt_params, ckpt_path):
    print("=========================== start converting rpnhead ===========================")
    new_params_list = []
    for ms_param in ms_params.keys():
        if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape and 'bbox_head' not in ms_param  :
            print(ms_param)
            ms_value = pt_params[ms_param]
            new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
    new_ckpt_path = 'C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.rpnhead.ckpt'
    ms.save_checkpoint(new_params_list, new_ckpt_path)

def bboxhead_param_convert(ms_params, pt_params, ckpt_path):
    print("=========================== start converting bboxhead ===========================")
    new_params_list = []
    param_dict = exchange_dict_keys_values(bbox_head_param_dic)
    for ms_param in ms_params.keys():
        if ms_param in param_dict.keys():  # and pt_params[param_dict[ms_param]].shape == ms_params[ms_param].shape:
            print(ms_param)
            ms_value = pt_params[param_dict[ms_param]]
            if ms_param == 'bbox_head.bbox_score.weight' or ms_param == 'bbox_head.bbox_delta.weight':
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value).transpose()})
            else:
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})  # dense 需要transpose
    new_ckpt_path = 'C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.bboxhead.ckpt'
    ms.save_checkpoint(new_params_list, new_ckpt_path)

def backbone_param_convert(ms_params, pt_params):
    print("=========================== start converting backbone ===========================")
    param_dict = exchange_dict_keys_values(backbone_param_dic)
    new_params_list = []
    for ms_param in param_dict.keys():
        if 'backbone' in ms_param:
            if ms_params[ms_param].shape == pt_params[param_dict[ms_param]].shape:
                print(ms_param, 'ok')
                ms_value = pt_params[param_dict[ms_param]]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, 'not ok!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    new_ckpt_path = 'C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.backbone.ckpt'
    ms.save_checkpoint(new_params_list, new_ckpt_path)

def exchange_dict_keys_values(x):
    new_dic = {}
    for key, val in x.items():
        new_dic[val] = key
    return new_dic



if __name__ == '__main__':
    # backbone = mindcv.create_model('resnet50', pretrained=False)
    # rpn_head = RPNHead()
    # bbox_assigner = BBoxAssigner
    # head = Res5Head(depth=50)
    # bbox_head = BboxHead(head=head, bbox_assigner=bbox_assigner, in_channel=2048)
    # fasterrcnn = FasterRCNN(backbone, rpn_head, bbox_head, None)

    fasterrcnn = build_fasterrcnn_model()

    # resnet50_for_fasterrcnn = build_resnet50_for_fasterrcnn()

    ckpt_path = 'C:\\tongli\\02workspace\\PaddleDetection\\weights\\faster_rcnn_r50_1x_coco.pdparams'


    pd_params = paddle_params(ckpt_path)
    print('='*40)
    ms_params = mindspore_params(fasterrcnn)

    print()
    backbone_param_convert(ms_params, pd_params)
    # rpnhead_param_convert(ms_params, pd_params, ckpt_path)
    # bboxhead_param_convert(ms_params, pd_params, ckpt_path)
