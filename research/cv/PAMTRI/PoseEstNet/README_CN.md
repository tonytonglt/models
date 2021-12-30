# 目录

<!-- TOC -->

- [目录](#目录)
- [PoseEstNet描述](#PoseEstNet描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [加载预训练权重](加载预训练权重)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [VeRi上的PoseEstNet](#VeRi上的PoseEstNet)
        - [推理性能](#推理性能)
            - [VeRi上的PoseEstNet](#VeRi上的PoseEstNet)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# PoseEstNet描述

姿势估计网络(PoseEstNet)是高分辨率网络（HRNet）的延伸，用于预测关键点坐标（有置信度/可见度）和生成热图/分段。

# 数据集

使用的数据集：[VeRi](https://vehiclereid.github.io/VeRi/)

- 数据集大小：961M，共776个id、5万张彩色图像
    - 训练集：9068张图像
    - 测试集：1060张图像
- 数据格式：RGB
    - 注：数据将在src/dataset/dataset.py中处理。

- 目录结构如下：

    ```bash
    ├── data
        ├── images
        |   ├── image_test
        |   |   ├── 0002_c002_00030600_0.jpg
        |   |   ├── 0002_c002_00030605_1.jpg
        |   |   ├── ...
        |   ├── image_train
        |   |   ├── 0001_c001_00016450_0.jpg
        |   |   ├── 0001_c001_00016460_0.jpg
        |   |   ├── ...
        ├── annot
        |   ├── label_test.csv
        |   ├── label_train.csv
        |   ├── image_train.json
        |   ├── image_test.json
    ```

其中[data/annot](data/annot)中内容已提供

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash
# 安装依赖包
pip install -r requirements.txt
```

安装完依赖包后，可以将下载后的VeRi数据集中`image_train/`文件下的图片复制到`data/images/image_train`和`data/images/image_test`中

```bash
# 运行训练示例
bash run_single_train.sh [DATA_PATH] [PRETRAINED_PATH] [DEVICE_ID]
# example: bash run_single_train.sh ../data/ /path/pretrained_path 0

# 运行分布式训练示例
bash run_distribute_train.sh [DATA_PATH] [pretrain_path] [RANK_TABLE]
# example: bash run_distribute_train.sh ../data/ /path/pretrain_path /path/rank_table

# 运行评估示例
bash run_eval.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID]
# example: bash run_eval.sh ../data/ /path/ckpt 0

# 运行转换数据示例
# 保留精度最高的ckpt, 然后运行trans生成MultiTaskNet所需要的数据集
# DATA_PATH 应该是MuitiTaskNet的数据集, 文件结构在MultiTaskNet中描述
bash run_trans.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID]
# example: bash run_trans.sh ../../MultiTaskNet/data/ /path/ckpt 0
```

对于分布式训练，需要提前创建JSON格式的hccl配置文件。
请遵循以下链接中的说明：
<https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

# 脚本说明

## 脚本及样例代码

```bash
├── cv
    ├── PAMTRI
        ├── README.md                   // PAMTRI描述
        ├── PoseEstNet
            ├── ascend_310_infer                    // 实现310推理源代码
            ├── scripts
            |   ├── run_single_train.sh             // 单卡到Ascend的shell脚本
            |   ├── run_distribute_train.sh         // 分布式到Ascend的shell脚本
            |   ├── run_eval.sh                     // Ascend评估的shell脚本
            |   ├── run_trans.sh                    // 生成数据集脚本
            |   ├── run_infer_310.sh                // Ascend推理shell脚本
            ├── src
            |   ├── config
            |   |   ├── default.py                  // 训练、推理参数
            |   |   ├── models.py                   // 模型参数
            |   ├── dataset
            |   |   ├── dataset.py                  // 数据处理
            |   |   ├── dataTrans.py                // 转换数据处理
            |   |   ├── evaluate.py                 // 推理acc计算
            |   |   ├── inference.py                // 标签信息
            |   |   ├── JointsDataset.py            // 基础数据处理
            |   |   ├── transform.py                // 转换输入数据
            |   |   ├── veri.py                     // veri数据集数据处理
            |   |   ├── zipreader.py                // 读取zip信息
            |   ├── loss
            |   |   ├── loss.py                     // 损失函数
            |   ├── model
            |   |   ├── HRNet.py                    // 模型
            |   ├── scheduler
            |   |   ├── lr.py                           // 学习率
            |   ├── utils
            |   |   ├── function.py                 // 推理函数
            |   |   ├── grid.py                     // 部分缺失算子实现
            |   |   ├── inference.py                // 标签信息
            |   |   ├── pthtockpt.py                // pth格式的预训练参数转换为ckpt
            |   |   ├── transform.py                // 转换数据集函数
            |   |   ├── vis.py                      // 转换数据集函数
            ├── config.yaml                         // 固定参数
            ├── eval.py                             // 精度验证脚本
            ├── train.py                            // 训练脚本
            ├── trans.py                            // 转换数据集脚本
            ├── export.py                           // 推理模型导出脚本
            ├── preprocess.py                       // 310推理前处理脚本
            ├── postprocess.py                      // 310推理后处理脚本
            ├── README.md                           // PoseEstNet描述
        ├── MultiTaskNet
```

## 脚本参数

```bash
--device_target:运行代码的设备, 默认为"Ascend"
--distribute:是否进行分布式训练
--data_dir:数据集所在路径
--pre_trained:是否要预训练
--pre_ckpt_path:预训练路径
--cfg:模型参数config.yaml路径
--ckpt_path:推理所需要的ckpt所在路径
```

## 训练过程

### 加载预训练权重

pytorch的HRNet预训练模型, [点击获取](https://drive.google.com/file/d/1aTXmxKAJVLsXbvM-TmQ0ZjJxP868G73q/view)

```bash
python pth2ckpt.py --pth_path /path/hrnet_w32-36af842e.pth
```

### 训练

- Ascend处理器环境运行

  ```bash
  bash run_single_train.sh ../data/ /path/pretrained_path 0
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train.log
  epoch:1 step:, loss is 2.4842823
  epcoh:2 step:, loss is 3.0897788
  ...
  ```

  模型检查点保存在当前目录下。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash run_distribute_train.sh ../data/ /path/pretrain_path /path/rank_table
  ```

  上述shell脚本将在后台运行分布训练。您可以通过`device[X]/train.log`文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "loss is" device0/train0.log
  epoch: 1 step: 70, loss is 1252.768
  epoch: 2 step: 70, loss is 757.1177
  ...
  # grep "loss is" device1/train1.log
  epoch: 1 step: 70, loss is 1293.7272
  epoch: 2 step: 70, loss is 746.15564
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估VeRi数据集

  ```bash
  bash run_eval.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID]
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  | Arch | Wheel | Fender | Back | Front | WindshieldBack | WindshieldFront | Mean | Mean@0.1 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | pose_hrnet | 85.334 | 81.334 | 70.501 | 76.727 | 86.651 | 89.610 | 82.324 | 16.704 |
  ```

## 导出过程

### 导出

将checkpoint文件导出成mindir格式模型。

```bash
python3.7 export.py --cfg config.yaml --ckpt_path CKPT_PATH
```

## 推理过程

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出。以下展示了使用mindir模型执行推理的示例。

- 在昇腾310上使用VeRi数据集进行推理

    执行推理的命令如下所示，其中'MINDIR_PATH'是mindir文件路径；'DATASET_PATH'是推理数据集路径；'NEED_PREPROCESS'表示数据集是否需要预处理，一般选择'y'；'DEVICE_TARGET'是设备类型，默认'Ascend';'DEVICE_ID'可选，默认值为0。

    ```bash
    bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
    ```

    推理的精度结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的分类准确率结果。

    ```bash
    | Arch | Wheel | Fender | Back | Front | WindshieldBack | WindshieldFront | Mean | Mean@0.1 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | pose_hrnet | 86.541 | 82.190 | 70.321 | 77.567 | 86.134 | 89.679 | 82.769 | 16.540 |
    ```

# 模型描述

## 性能

### 评估性能

#### VeRi上的PoseEstNet

| 参数                 | Ascend                    |
| -------------------- | ------------------------- |
| 模型版本              | PoseEstNet             |
| 资源                  | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8                |
| 上传日期              | 2021-9-30                 |
| MindSpore版本         | 1.3.0            |
| 数据集                | VeRi             |
| 训练参数              | epoch=210, batch_size = 16(单卡可以为32), lr=0.001  |
| 优化器                | adam                 |
| 损失函数              | JointsMSELoss             |
| 输出                  | 概率                      |
| 损失                       | 92.459                                                     |
| 速度                  | 8卡：130毫秒/步 |
| 总时长                | 8卡：40分钟 |
| 微调检查点 | 335M (.ckpt文件)                                         |
| 推理模型        | 113M (.mindir文件)                     |
| 脚本                    | [PoseEstNet脚本](https://gitee.com/kqzhang/models/tree/master/research/cv/PAMTRI/PoseEstNet) |

### 推理性能

#### VeRi上的PoseEstNet

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本       | Inception V1                |
| 资源            |  Ascend 910；系统 Euler2.8                  |
| 上传日期       | 2021-09-30 |
| MindSpore 版本   | 1.3.0                       |
| 数据集             | VeRi     |
| batch_size          | 32                         |
| 输出             | 概率                 |
| 准确性            | 82.324   |
| 推理模型 | 113M (.mindir文件)         |

# 随机情况说明

dataset.py中使用随机选择策略。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。