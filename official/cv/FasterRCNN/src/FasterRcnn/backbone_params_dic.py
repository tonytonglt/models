backbone_param_dic = {
    "backbone.conv1.conv1.conv.weight": "backbone.conv1.weight",
    "backbone.conv1.conv1.norm._mean": "backbone.bn1.moving_mean",
    "backbone.conv1.conv1.norm._variance": "backbone.bn1.moving_variance",
    "backbone.conv1.conv1.norm.bias": "backbone.bn1.beta",
    "backbone.conv1.conv1.norm.weight": "backbone.bn1.gamma",

    "backbone.res2.res2a.branch2a.conv.weight": "backbone.layer1.0.conv1.weight",
    "backbone.res2.res2a.branch2a.norm._mean": "backbone.layer1.0.bn1.moving_mean",
    "backbone.res2.res2a.branch2a.norm._variance":"backbone.layer1.0.bn1.moving_variance",
    "backbone.res2.res2a.branch2a.norm.bias": "backbone.layer1.0.bn1.beta",
    "backbone.res2.res2a.branch2a.norm.weight": "backbone.layer1.0.bn1.gamma",
    "backbone.res2.res2a.branch2b.conv.weight": "backbone.layer1.0.conv2.weight",
    "backbone.res2.res2a.branch2b.norm._mean": "backbone.layer1.0.bn2.moving_mean",
    "backbone.res2.res2a.branch2b.norm._variance": "backbone.layer1.0.bn2.moving_variance",
    "backbone.res2.res2a.branch2b.norm.bias": "backbone.layer1.0.bn2.beta",
    "backbone.res2.res2a.branch2b.norm.weight": "backbone.layer1.0.bn2.gamma",
    "backbone.res2.res2a.branch2c.conv.weight": "backbone.layer1.0.conv3.weight",
    "backbone.res2.res2a.branch2c.norm._mean": "backbone.layer1.0.bn3.moving_mean",
    "backbone.res2.res2a.branch2c.norm._variance": "backbone.layer1.0.bn3.moving_variance",
    "backbone.res2.res2a.branch2c.norm.bias": "backbone.layer1.0.bn3.beta",
    "backbone.res2.res2a.branch2c.norm.weight": "backbone.layer1.0.bn3.gamma",
    "backbone.res2.res2a.short.conv.weight": "backbone.layer1.0.down_sample.0.weight",
    "backbone.res2.res2a.short.norm._mean": "backbone.layer1.0.down_sample.1.moving_mean",
    "backbone.res2.res2a.short.norm._variance": "backbone.layer1.0.down_sample.1.moving_variance",
    "backbone.res2.res2a.short.norm.bias": "backbone.layer1.0.down_sample.1.beta",
    "backbone.res2.res2a.short.norm.weight": "backbone.layer1.0.down_sample.1.gamma",

    "backbone.res2.res2b.branch2a.conv.weight": "backbone.layer1.1.conv1.weight",
    "backbone.res2.res2b.branch2a.norm._mean": "backbone.layer1.1.bn1.moving_mean",
    "backbone.res2.res2b.branch2a.norm._variance": "backbone.layer1.1.bn1.moving_variance",
    "backbone.res2.res2b.branch2a.norm.bias": "backbone.layer1.1.bn1.beta",
    "backbone.res2.res2b.branch2a.norm.weight": "backbone.layer1.1.bn1.gamma",
    "backbone.res2.res2b.branch2b.conv.weight": "backbone.layer1.1.conv2.weight",
    "backbone.res2.res2b.branch2b.norm._mean": "backbone.layer1.1.bn2.moving_mean",
    "backbone.res2.res2b.branch2b.norm._variance": "backbone.layer1.1.bn2.moving_variance",
    "backbone.res2.res2b.branch2b.norm.bias": "backbone.layer1.1.bn2.beta",
    "backbone.res2.res2b.branch2b.norm.weight": "backbone.layer1.1.bn2.gamma",
    "backbone.res2.res2b.branch2c.conv.weight": "backbone.layer1.1.conv3.weight",
    "backbone.res2.res2b.branch2c.norm._mean": "backbone.layer1.1.bn3.moving_mean",
    "backbone.res2.res2b.branch2c.norm._variance": "backbone.layer1.1.bn3.moving_variance",
    "backbone.res2.res2b.branch2c.norm.bias": "backbone.layer1.1.bn3.beta",
    "backbone.res2.res2b.branch2c.norm.weight": "backbone.layer1.1.bn3.gamma",

    "backbone.res2.res2c.branch2a.conv.weight": "backbone.layer1.2.conv1.weight",
    "backbone.res2.res2c.branch2a.norm._mean": "backbone.layer1.2.bn1.moving_mean",
    "backbone.res2.res2c.branch2a.norm._variance": "backbone.layer1.2.bn1.moving_variance",
    "backbone.res2.res2c.branch2a.norm.bias": "backbone.layer1.2.bn1.beta",
    "backbone.res2.res2c.branch2a.norm.weight": "backbone.layer1.2.bn1.gamma",
    "backbone.res2.res2c.branch2b.conv.weight": "backbone.layer1.2.conv2.weight",
    "backbone.res2.res2c.branch2b.norm._mean": "backbone.layer1.2.bn2.moving_mean",
    "backbone.res2.res2c.branch2b.norm._variance": "backbone.layer1.2.bn2.moving_variance",
    "backbone.res2.res2c.branch2b.norm.bias": "backbone.layer1.2.bn2.beta",
    "backbone.res2.res2c.branch2b.norm.weight": "backbone.layer1.2.bn2.gamma",
    "backbone.res2.res2c.branch2c.conv.weight": "backbone.layer1.2.conv3.weight",
    "backbone.res2.res2c.branch2c.norm._mean": "backbone.layer1.2.bn3.moving_mean",
    "backbone.res2.res2c.branch2c.norm._variance": "backbone.layer1.2.bn3.moving_variance",
    "backbone.res2.res2c.branch2c.norm.bias": "backbone.layer1.2.bn3.beta",
    "backbone.res2.res2c.branch2c.norm.weight": "backbone.layer1.2.bn3.gamma",

    "backbone.res3.res3a.branch2a.conv.weight": "backbone.layer2.0.conv1.weight",
    "backbone.res3.res3a.branch2a.norm._mean": "backbone.layer2.0.bn1.moving_mean",
    "backbone.res3.res3a.branch2a.norm._variance": "backbone.layer2.0.bn1.moving_variance",
    "backbone.res3.res3a.branch2a.norm.bias": "backbone.layer2.0.bn1.beta",
    "backbone.res3.res3a.branch2a.norm.weight": "backbone.layer2.0.bn2.gamma",
    "backbone.res3.res3a.branch2b.conv.weight": "backbone.layer2.0.conv2.weight",
    "backbone.res3.res3a.branch2b.norm._mean": "backbone.layer2.0.bn2.moving_mean",
    "backbone.res3.res3a.branch2b.norm._variance": "backbone.layer2.0.bn2.moving_variance",
    "backbone.res3.res3a.branch2b.norm.bias": "backbone.layer2.0.bn2.beta",
    "backbone.res3.res3a.branch2b.norm.weight": "backbone.layer2.0.bn1.gamma",
    "backbone.res3.res3a.branch2c.conv.weight": "backbone.layer2.0.conv3.weight",
    "backbone.res3.res3a.branch2c.norm._mean": "backbone.layer2.0.bn3.moving_mean",
    "backbone.res3.res3a.branch2c.norm._variance": "backbone.layer2.0.bn3.moving_variance",
    "backbone.res3.res3a.branch2c.norm.bias": "backbone.layer2.0.bn3.beta",
    "backbone.res3.res3a.branch2c.norm.weight": "backbone.layer2.0.bn3.gamma",
    "backbone.res3.res3a.short.conv.weight": "backbone.layer2.0.down_sample.0.weight",
    "backbone.res3.res3a.short.norm._mean": "backbone.layer2.0.down_sample.1.moving_mean",
    "backbone.res3.res3a.short.norm._variance": "backbone.layer2.0.down_sample.1.moving_variance",
    "backbone.res3.res3a.short.norm.bias": "backbone.layer2.0.down_sample.1.beta",
    "backbone.res3.res3a.short.norm.weight": "backbone.layer2.0.down_sample.1.gamma",

    "backbone.res3.res3b.branch2a.conv.weight": "backbone.layer2.1.conv1.weight",
    "backbone.res3.res3b.branch2a.norm._mean": "backbone.layer2.1.bn1.moving_mean",
    "backbone.res3.res3b.branch2a.norm._variance": "backbone.layer2.1.bn1.moving_variance",
    "backbone.res3.res3b.branch2a.norm.bias": "backbone.layer2.1.bn1.beta",
    "backbone.res3.res3b.branch2a.norm.weight": "backbone.layer2.1.bn1.gamma",

    "backbone.res3.res3b.branch2b.conv.weight": "backbone.layer2.1.conv2.weight",
    "backbone.res3.res3b.branch2b.norm._mean": "backbone.layer2.1.bn2.moving_mean",
    "backbone.res3.res3b.branch2b.norm._variance": "backbone.layer2.1.bn2.moving_variance",
    "backbone.res3.res3b.branch2b.norm.bias": "backbone.layer2.1.bn2.beta",
    "backbone.res3.res3b.branch2b.norm.weight": "backbone.layer2.1.bn2.gamma",

    "backbone.res3.res3b.branch2c.conv.weight": "backbone.layer2.1.conv3.weight",
    "backbone.res3.res3b.branch2c.norm._mean": "backbone.layer2.1.bn3.moving_mean",
    "backbone.res3.res3b.branch2c.norm._variance": "backbone.layer2.1.bn3.moving_variance",
    "backbone.res3.res3b.branch2c.norm.bias": "backbone.layer2.1.bn3.beta",
    "backbone.res3.res3b.branch2c.norm.weight": "backbone.layer2.1.bn3.gamma",

    "backbone.res3.res3c.branch2a.conv.weight": "backbone.layer2.2.conv1.weight",
    "backbone.res3.res3c.branch2a.norm._mean": "backbone.layer2.2.bn1.moving_mean",
    "backbone.res3.res3c.branch2a.norm._variance": "backbone.layer2.2.bn1.moving_variance",
    "backbone.res3.res3c.branch2a.norm.bias": "backbone.layer2.2.bn1.beta",
    "backbone.res3.res3c.branch2a.norm.weight": "backbone.layer2.2.bn1.gamma",

    "backbone.res3.res3c.branch2b.conv.weight": "backbone.layer2.2.conv2.weight",
    "backbone.res3.res3c.branch2b.norm._mean": "backbone.layer2.2.bn2.moving_mean",
    "backbone.res3.res3c.branch2b.norm._variance": "backbone.layer2.2.bn2.moving_variance",
    "backbone.res3.res3c.branch2b.norm.bias": "backbone.layer2.2.bn2.beta",
    "backbone.res3.res3c.branch2b.norm.weight": "backbone.layer2.2.bn2.gamma",

    "backbone.res3.res3c.branch2c.conv.weight": "backbone.layer2.2.conv3.weight",
    "backbone.res3.res3c.branch2c.norm._mean": "backbone.layer2.2.bn3.moving_mean",
    "backbone.res3.res3c.branch2c.norm._variance": "backbone.layer2.2.bn3.moving_variance",
    "backbone.res3.res3c.branch2c.norm.bias": "backbone.layer2.2.bn3.beta",
    "backbone.res3.res3c.branch2c.norm.weight": "backbone.layer2.2.bn3.gamma",

    "backbone.res3.res3d.branch2a.conv.weight": "backbone.layer2.3.conv1.weight",
    "backbone.res3.res3d.branch2a.norm._mean": "backbone.layer2.3.bn1.moving_mean",
    "backbone.res3.res3d.branch2a.norm._variance": "backbone.layer2.3.bn1.moving_variance",
    "backbone.res3.res3d.branch2a.norm.bias": "backbone.layer2.3.bn1.beta",
    "backbone.res3.res3d.branch2a.norm.weight": "backbone.layer2.3.bn1.gamma",

    "backbone.res3.res3d.branch2b.conv.weight": "backbone.layer2.3.conv2.weight",
    "backbone.res3.res3d.branch2b.norm._mean": "backbone.layer2.3.bn2.moving_mean",
    "backbone.res3.res3d.branch2b.norm._variance": "backbone.layer2.3.bn2.moving_variance",
    "backbone.res3.res3d.branch2b.norm.bias": "backbone.layer2.3.bn2.beta",
    "backbone.res3.res3d.branch2b.norm.weight": "backbone.layer2.3.bn2.gamma",

    "backbone.res3.res3d.branch2c.conv.weight": "backbone.layer2.3.conv3.weight",
    "backbone.res3.res3d.branch2c.norm._mean": "backbone.layer2.3.bn3.moving_mean",
    "backbone.res3.res3d.branch2c.norm._variance": "backbone.layer2.3.bn3.moving_variance",
    "backbone.res3.res3d.branch2c.norm.bias": "backbone.layer2.3.bn3.beta",
    "backbone.res3.res3d.branch2c.norm.weight": "backbone.layer2.3.bn3.gamma",

    "backbone.res4.res4a.branch2a.conv.weight": "backbone.layer3.0.conv1.weight",
    "backbone.res4.res4a.branch2a.norm._mean": "backbone.layer3.0.bn1.moving_mean",
    "backbone.res4.res4a.branch2a.norm._variance": "backbone.layer3.0.bn1.moving_variance",
    "backbone.res4.res4a.branch2a.norm.bias": "backbone.layer3.0.bn1.beta",
    "backbone.res4.res4a.branch2a.norm.weight": "backbone.layer3.0.bn1.gamma",
    "backbone.res4.res4a.branch2b.conv.weight": "backbone.layer3.0.conv2.weight",
    "backbone.res4.res4a.branch2b.norm._mean": "backbone.layer3.0.bn2.moving_mean",
    "backbone.res4.res4a.branch2b.norm._variance": "backbone.layer3.0.bn2.moving_variance",
    "backbone.res4.res4a.branch2b.norm.bias": "backbone.layer3.0.bn2.beta",
    "backbone.res4.res4a.branch2b.norm.weight": "backbone.layer3.0.bn2.gamma",
    "backbone.res4.res4a.branch2c.conv.weight": "backbone.layer3.0.conv3.weight",
    "backbone.res4.res4a.branch2c.norm._mean": "backbone.layer3.0.bn3.moving_mean",
    "backbone.res4.res4a.branch2c.norm._variance": "backbone.layer3.0.bn3.moving_variance",
    "backbone.res4.res4a.branch2c.norm.bias": "backbone.layer3.0.bn3.beta",
    "backbone.res4.res4a.branch2c.norm.weight": "backbone.layer3.0.bn3.gamma",
    "backbone.res4.res4a.short.conv.weight": "backbone.layer3.0.down_sample.0.weight",
    "backbone.res4.res4a.short.norm._mean": "backbone.layer3.0.down_sample.1.moving_mean",
    "backbone.res4.res4a.short.norm._variance": "backbone.layer3.0.down_sample.1.moving_variance",
    "backbone.res4.res4a.short.norm.bias": "backbone.layer3.0.down_sample.1.beta",
    "backbone.res4.res4a.short.norm.weight": "backbone.layer3.0.down_sample.1.gamma",

    "backbone.res4.res4b.branch2a.conv.weight": "backbone.layer3.1.conv1.weight",
    "backbone.res4.res4b.branch2a.norm._mean": "backbone.layer3.1.bn1.moving_mean",
    "backbone.res4.res4b.branch2a.norm._variance": "backbone.layer3.1.bn1.moving_variance",
    "backbone.res4.res4b.branch2a.norm.bias": "backbone.layer3.1.bn1.beta",
    "backbone.res4.res4b.branch2a.norm.weight": "backbone.layer3.1.bn1.gamma",

    "backbone.res4.res4b.branch2b.conv.weight": "backbone.layer3.1.conv2.weight",
    "backbone.res4.res4b.branch2b.norm._mean": "backbone.layer3.1.bn2.moving_mean",
    "backbone.res4.res4b.branch2b.norm._variance": "backbone.layer3.1.bn2.moving_variance",
    "backbone.res4.res4b.branch2b.norm.bias": "backbone.layer3.1.bn2.beta",
    "backbone.res4.res4b.branch2b.norm.weight": "backbone.layer3.1.bn2.gamma",

    "backbone.res4.res4b.branch2c.conv.weight": "backbone.layer3.1.conv3.weight",
    "backbone.res4.res4b.branch2c.norm._mean": "backbone.layer3.1.bn3.moving_mean",
    "backbone.res4.res4b.branch2c.norm._variance": "backbone.layer3.1.bn3.moving_variance",
    "backbone.res4.res4b.branch2c.norm.bias": "backbone.layer3.1.bn3.beta",
    "backbone.res4.res4b.branch2c.norm.weight": "backbone.layer3.1.bn3.gamma",

    "backbone.res4.res4c.branch2a.conv.weight": "backbone.layer3.2.conv1.weight",
    "backbone.res4.res4c.branch2a.norm._mean": "backbone.layer3.2.bn1.moving_mean",
    "backbone.res4.res4c.branch2a.norm._variance": "backbone.layer3.2.bn1.moving_variance",
    "backbone.res4.res4c.branch2a.norm.bias": "backbone.layer3.2.bn1.beta",
    "backbone.res4.res4c.branch2a.norm.weight": "backbone.layer3.2.bn1.gamma",

    "backbone.res4.res4c.branch2b.conv.weight": "backbone.layer3.2.conv2.weight",
    "backbone.res4.res4c.branch2b.norm._mean": "backbone.layer3.2.bn2.moving_mean",
    "backbone.res4.res4c.branch2b.norm._variance": "backbone.layer3.2.bn2.moving_variance",
    "backbone.res4.res4c.branch2b.norm.bias": "backbone.layer3.2.bn2.beta",
    "backbone.res4.res4c.branch2b.norm.weight": "backbone.layer3.2.bn2.gamma",

    "backbone.res4.res4c.branch2c.conv.weight": "backbone.layer3.2.conv3.weight",
    "backbone.res4.res4c.branch2c.norm._mean": "backbone.layer3.2.bn3.moving_mean",
    "backbone.res4.res4c.branch2c.norm._variance": "backbone.layer3.2.bn3.moving_variance",
    "backbone.res4.res4c.branch2c.norm.bias": "backbone.layer3.2.bn3.beta",
    "backbone.res4.res4c.branch2c.norm.weight": "backbone.layer3.2.bn3.gamma",

    "backbone.res4.res4d.branch2a.conv.weight": "backbone.layer3.3.conv1.weight",
    "backbone.res4.res4d.branch2a.norm._mean": "backbone.layer3.3.bn1.moving_mean",
    "backbone.res4.res4d.branch2a.norm._variance": "backbone.layer3.3.bn1.moving_variance",
    "backbone.res4.res4d.branch2a.norm.bias": "backbone.layer3.3.bn1.beta",
    "backbone.res4.res4d.branch2a.norm.weight": "backbone.layer3.3.bn1.gamma",

    "backbone.res4.res4d.branch2b.conv.weight": "backbone.layer3.3.conv2.weight",
    "backbone.res4.res4d.branch2b.norm._mean": "backbone.layer3.3.bn2.moving_mean",
    "backbone.res4.res4d.branch2b.norm._variance": "backbone.layer3.3.bn2.moving_variance",
    "backbone.res4.res4d.branch2b.norm.bias": "backbone.layer3.3.bn2.beta",
    "backbone.res4.res4d.branch2b.norm.weight": "backbone.layer3.3.bn2.gamma",

    "backbone.res4.res4d.branch2c.conv.weight": "backbone.layer3.3.conv3.weight",
    "backbone.res4.res4d.branch2c.norm._mean": "backbone.layer3.3.bn3.moving_mean",
    "backbone.res4.res4d.branch2c.norm._variance": "backbone.layer3.3.bn3.moving_variance",
    "backbone.res4.res4d.branch2c.norm.bias": "backbone.layer3.3.bn3.beta",
    "backbone.res4.res4d.branch2c.norm.weight": "backbone.layer3.3.bn3.gamma",

    "backbone.res4.res4e.branch2a.conv.weight": "backbone.layer3.4.conv1.weight",
    "backbone.res4.res4e.branch2a.norm._mean": "backbone.layer3.4.bn1.moving_mean",
    "backbone.res4.res4e.branch2a.norm._variance": "backbone.layer3.4.bn1.moving_variance",
    "backbone.res4.res4e.branch2a.norm.bias": "backbone.layer3.4.bn1.beta",
    "backbone.res4.res4e.branch2a.norm.weight": "backbone.layer3.4.bn1.gamma",

    "backbone.res4.res4e.branch2b.conv.weight": "backbone.layer3.4.conv2.weight",
    "backbone.res4.res4e.branch2b.norm._mean": "backbone.layer3.4.bn2.moving_mean",
    "backbone.res4.res4e.branch2b.norm._variance": "backbone.layer3.4.bn2.moving_variance",
    "backbone.res4.res4e.branch2b.norm.bias": "backbone.layer3.4.bn2.beta",
    "backbone.res4.res4e.branch2b.norm.weight": "backbone.layer3.4.bn2.gamma",

    "backbone.res4.res4e.branch2c.conv.weight": "backbone.layer3.4.conv3.weight",
    "backbone.res4.res4e.branch2c.norm._mean": "backbone.layer3.4.bn3.moving_mean",
    "backbone.res4.res4e.branch2c.norm._variance": "backbone.layer3.4.bn3.moving_variance",
    "backbone.res4.res4e.branch2c.norm.bias": "backbone.layer3.4.bn3.beta",
    "backbone.res4.res4e.branch2c.norm.weight": "backbone.layer3.4.bn3.gamma",

    "backbone.res4.res4f.branch2a.conv.weight": "backbone.layer3.5.conv1.weight",
    "backbone.res4.res4f.branch2a.norm._mean": "backbone.layer3.5.bn1.moving_mean",
    "backbone.res4.res4f.branch2a.norm._variance": "backbone.layer3.5.bn1.moving_variance",
    "backbone.res4.res4f.branch2a.norm.bias": "backbone.layer3.5.bn1.beta",
    "backbone.res4.res4f.branch2a.norm.weight": "backbone.layer3.5.bn1.gamma",

    "backbone.res4.res4f.branch2b.conv.weight": "backbone.layer3.5.conv2.weight",
    "backbone.res4.res4f.branch2b.norm._mean": "backbone.layer3.5.bn2.moving_mean",
    "backbone.res4.res4f.branch2b.norm._variance": "backbone.layer3.5.bn2.moving_variance",
    "backbone.res4.res4f.branch2b.norm.bias": "backbone.layer3.5.bn2.beta",
    "backbone.res4.res4f.branch2b.norm.weight": "backbone.layer3.5.bn2.gamma",

    "backbone.res4.res4f.branch2c.conv.weight": "backbone.layer3.5.conv3.weight",
    "backbone.res4.res4f.branch2c.norm._mean": "backbone.layer3.5.bn3.moving_mean",
    "backbone.res4.res4f.branch2c.norm._variance": "backbone.layer3.5.bn3.moving_variance",
    "backbone.res4.res4f.branch2c.norm.bias": "backbone.layer3.5.bn3.beta",
    "backbone.res4.res4f.branch2c.norm.weight": "backbone.layer3.5.bn3.gamma",
}

