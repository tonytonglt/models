import mindspore as ms
from mindspore import ops, nn
from mindspore.amp import StaticLossScaler, all_finite
from mindspore.communication.management import init, get_rank, get_group_size

from src.FasterRcnn.build_fasterrcnn import build_fasterrcnn_model
from src.data.loader import build_dataloader



def init_env(cfg):
    """初始化运行时环境."""
    ms.set_seed(cfg.seed)
    # 如果device_target设置是None，利用框架自动获取device_target，否则使用设置的。
    if cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg.device_target)

    # 配置运行模式，支持图模式和PYNATIVE模式
    if cfg.context_mode not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative']")
    # context_mode = ms.GRAPH_MODE if cfg.context_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

    cfg.device_target = ms.get_context("device_target")
    # 如果是CPU上运行的话，不配置多卡环境
    if cfg.device_target == "CPU":
        cfg.device_id = 0
        cfg.device_num = 1
        cfg.rank_id = 0

    # 设置运行时使用的卡
    if hasattr(cfg, "device_id") and isinstance(cfg.device_id, int):
        ms.set_context(device_id=cfg.device_id)

    if cfg.device_num > 1:
        # init方法用于多卡的初始化，不区分Ascend和GPU，get_group_size和get_rank方法只能在init后使用
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        if cfg.device_num != group_size:
            raise ValueError(f"the setting device_num: {cfg.device_num} not equal to the real group_size: {group_size}")
        cfg.rank_id = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if hasattr(cfg, "all_reduce_fusion_config"):
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg.all_reduce_fusion_config)
    else:
        cfg.device_num = 1
        cfg.rank_id = 0
        print("run standalone!", flush=True)


class Trainer:
    """一个有两个loss的训练示例"""
    def __init__(self, net, train_dataset, loss_scale=1.0, eval_dataset=None, metric=None):
        self.net = net
        # self.loss1 = loss1
        # self.loss2 = loss2
        self.optimizer = nn.Adam(self.net.trainable_params())
        self.train_dataset = train_dataset
        # self.train_data_size = self.train_dataset.get_dataset_size()    # 获取训练集batch数
        self.train_data_size = 10000
        self.weights = self.optimizer.parameters
        # 注意value_and_grad的第一个参数需要是需要做梯度求导的图，一般包含网络和loss。这里可以是一个函数，也可以是Cell
        self.value_and_grad = ops.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)

        # 分布式场景使用
        self.grad_reducer = self.get_grad_reducer()
        self.loss_scale = StaticLossScaler(loss_scale)
        self.run_eval = eval_dataset is not None
        if self.run_eval:
            self.eval_dataset = eval_dataset
            self.metric = metric
            self.best_acc = 0

    def get_grad_reducer(self):
        grad_reducer = ops.identity
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        # 判断是否是分布式场景，分布式场景的设置参考上面通用运行环境设置
        reducer_flag = (parallel_mode != ms.ParallelMode.STAND_ALONE)
        if reducer_flag:
            grad_reducer = nn.DistributedGradReducer(self.weights)
        return grad_reducer

    def forward_fn(self, inputs):
        """正向网络构建，注意第一个输出必须是最后需要求梯度的那个输出"""
        # inputs_dict = {}
        inputs['image'] = inputs['image'].asnumpy()
        inputs['w'] = inputs['w'].asnumpy()
        inputs['h'] = inputs['h'].asnumpy()
        inputs['gt_bbox'] = inputs['gt_bbox'].asnumpy()
        inputs['gt_class'] = inputs['gt_class'].asnumpy()
        rpn_loss, bbox_loss = self.net(inputs)
        # loss1 = self.loss1(logits, labels)
        # loss2 = self.loss2(logits, labels)
        # loss1 = self.net.rpn_head.get_loss(pred_scores, pred_deltas, anchors, inputs)
        # loss2 = self.net.bbox_head.get_loss()
        loss_rpn_cls = rpn_loss['loss_rpn_cls']
        loss_rpn_reg = rpn_loss['loss_rpn_reg']
        loss_bbox_cls = bbox_loss['loss_bbox_cls']
        loss_bbox_reg = bbox_loss['loss_bbox_reg']
        loss = loss_rpn_cls + loss_rpn_reg + loss_bbox_cls + loss_bbox_reg
        # loss = self.loss_scale.scale(loss)
        return loss, loss_rpn_cls, loss_rpn_reg, loss_bbox_cls, loss_bbox_reg

    # @ms.jit    # jit加速，需要满足图模式构建的要求，否则会报错
    def train_single(self, inputs):
        # rpn_loss, bbox_loss = self.net(inputs)
        # inputs = list(inputs.values())
        inputs['image'] = ms.Tensor(inputs['image'], dtype=ms.int32)
        inputs['w'] = ms.Tensor(inputs['w'], dtype=ms.float32)
        inputs['h'] = ms.Tensor(inputs['h'], dtype=ms.float32)
        inputs['gt_bbox'] = ms.Tensor(inputs['gt_bbox'], dtype=ms.float32)
        inputs['gt_class'] = ms.Tensor(inputs['gt_class'], dtype=ms.int32)

        (loss, loss_rpn_cls, loss_rpn_reg, loss_bbox_cls, loss_bbox_reg), grads = self.value_and_grad(inputs)
        loss = self.loss_scale.unscale(loss)
        grads = self.loss_scale.unscale(grads)
        grads = self.grad_reducer(grads)
        state = all_finite(grads)
        if state:
            self.optimizer(grads)

        return loss, loss_rpn_cls, loss_rpn_reg, loss_bbox_cls, loss_bbox_reg

    def train(self, epochs):
        # train_dataset = self.train_dataset.create_dict_iterator()
        dataset_iterator = self.train_dataset
        self.net.set_train(True)
        for epoch in range(epochs):
            # 训练一个epoch
            for batch, data in enumerate(dataset_iterator):
                loss, loss_rpn_cls, loss_rpn_reg, loss_bbox_cls, loss_bbox_reg = self.train_single(data)
                if batch % 1 == 0:
                    print(f"step: [{batch} /{self.train_data_size}] "
                          f"loss: {loss}, loss_rpn_cls: {loss_rpn_cls}, loss_rpn_reg: {loss_rpn_reg}, "
                          f"loss_bbox_cls: {loss_bbox_cls}, loss_bbox_reg: {loss_bbox_reg}", flush=True)
            # 推理并保存最好的那个checkpoint
            if self.run_eval:
                eval_dataset = self.eval_dataset.create_dict_iterator(num_epochs=1)
                self.net.set_train(False)
                self.metric.clear()
                for batch, data in enumerate(eval_dataset):
                    output = self.net(data["image"])
                    self.metric.update(output, data["label"])
                accuracy = self.metric.eval()
                print(f"epoch {epoch}, accuracy: {accuracy}", flush=True)
                if accuracy >= self.best_acc:
                    # 保存最好的那个checkpoint
                    self.best_acc = accuracy
                    ms.save_checkpoint(self.net, "best.ckpt")
                    print(f"Updata best acc: {accuracy}")
                self.net.set_train(True)




def train():
    faster_rcnn = build_fasterrcnn_model()
    dataloader = build_dataloader()
    trainer = Trainer(faster_rcnn, dataloader)
    trainer.train(epochs=50)
    print()

if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)
    train()
