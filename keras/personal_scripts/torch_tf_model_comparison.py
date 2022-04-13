from keras.applications.resnet import ResNet
from keras import layers
import numpy as np
import tensorflow as tf
import torch
from torchvision import models


torch_resnet18 = models.resnet18(pretrained=True).eval()
torch_resnet34 = models.resnet34(pretrained=True).eval()


def basic_block(x, filters, stride=1, use_bias=True, conv_shortcut=True,
                name=None):
    """A basic residual block for ResNet18 and 34.

    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

    Returns:
    Output tensor for the basic residual block.
    """
    bn_axis = 3
    kernel_size = 3

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=use_bias,
            name=name + '_0_conv',
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
        filters, kernel_size, padding='SAME', strides=stride,
        use_bias=use_bias,
        name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='SAME',
        use_bias=use_bias,
        name=name + '_2_conv',
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x

def stack_block(
    x,
    filters,
    n_blocks,
    block_fn,
    stride1=2,
    first_shortcut=True,
    name=None,
    use_bias=False,
):
    """A set of stacked residual blocks.

    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    n_blocks: integer, blocks in the stacked blocks.
    block_fn: callable, function defining one block.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.

    Returns:
    Output tensor for the stacked basic blocks.
    """
    x = block_fn(
        x,
        filters,
        stride=stride1,
        conv_shortcut=first_shortcut,
        use_bias=use_bias,
        name=name + '_block1',
    )
    for i in range(2, n_blocks + 1):
        x = block_fn(
            x,
            filters,
            conv_shortcut=False,
            use_bias=use_bias,
            name=name + '_block' + str(i),
        )
    return x


def ResNet18(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             use_bias=True,
             **kwargs):
    """Instantiates the ResNet18 architecture."""

    def stack_fn(x):
        x = stack_block(
            x,
            64,
            2,
            basic_block,
            use_bias=use_bias,
            first_shortcut=False,
            stride1=1,
            name='conv2',
        )
        x = stack_block(
            x,
            128,
            2,
            basic_block,
            use_bias=use_bias,
            name='conv3',
        )
        x = stack_block(
            x,
            256,
            2,
            basic_block,
            use_bias=use_bias,
            name='conv4',
        )
        return stack_block(
            x,
            512,
            2,
            basic_block,
            use_bias=use_bias,
            name='conv5',
        )

    return ResNet(
        stack_fn,
        False,
        use_bias,
        'resnet18',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )


def ResNet34(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             use_bias=True,
             **kwargs):
    """Instantiates the ResNet34 architecture."""

    def stack_fn(x):
        x = stack_block(
            x,
            64,
            3,
            basic_block,
            use_bias=use_bias,
            first_shortcut=False,
            stride1=1,
            name='conv2',
        )
        x = stack_block(
            x,
            128,
            4,
            basic_block,
            use_bias=use_bias,
            name='conv3',
        )
        x = stack_block(
            x,
            256,
            6,
            basic_block,
            use_bias=use_bias,
            name='conv4',
        )
        return stack_block(
            x,
            512,
            3,
            basic_block,
            use_bias=use_bias,
            name='conv5',
        )

    return ResNet(
        stack_fn,
        False,
        use_bias,
        'resnet34',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )

tf_resnet18 = ResNet18(weights='resnet18.h5', use_bias=True, input_shape=(224, 224, 3))
tf_resnet34 = ResNet34(weights='resnet34.h5', use_bias=True, input_shape=(224, 224, 3))

for torch_model, tf_model in zip(
    [torch_resnet18, torch_resnet34],
    [tf_resnet18, tf_resnet34],
):
    np.random.seed(0)
    example_input = np.random.normal(size=(1, 224, 224, 3))
    tf_input = tf.constant(example_input, dtype=tf.float32)
    torch_input = torch.from_numpy(np.transpose(example_input, (0, 3, 1, 2))).float()
    tf_output = tf_model(tf_input, training=False)
    torch_output = torch.softmax(torch_model(torch_input), dim=1)

    #### Unique conv testing #####
    example_input = np.random.normal(size=(1, 224, 224, 64))
    tf_input = tf.constant(example_input, dtype=tf.float32)
    torch_input = torch.from_numpy(np.transpose(example_input, (0, 3, 1, 2))).float()
    torch_conv = torch_model.layer2[0].conv1
    tf_conv = tf_model.get_layer('conv3_block1_1_conv')
    tf_conv_output = tf_conv(tf_input, training=False)
    torch_conv_output = torch_conv(torch_input)
    np.testing.assert_equal(
        tf_conv.get_weights()[1],
        np.zeros_like(tf_conv.get_weights()[1]),
    )
    print('bias is 0')
    np.testing.assert_almost_equal(
        np.transpose(tf_conv.get_weights()[0], (3, 2, 0, 1)),
        torch_conv.weight.data.numpy(),
    )
    print('weights are the same')
    import pdb; pdb.set_trace()
    np.testing.assert_almost_equal(
        np.transpose(tf_conv_output.numpy(), (0, 3, 1, 2)),
        torch_conv_output.detach().numpy(),
    )




    def torch_inter_model(inputs):
        x = torch_model.conv1(inputs)
        x = torch_model.bn1(x)
        x = torch_model.relu(x)
        x = torch_model.maxpool(x)

        x = torch_model.layer1(x)
        x = torch_model.layer2[0].conv1(x)
        # x = torch_model.layer3(x)
        # x = torch_model.layer4(x)

        # x = torch_model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = torch_model.fc(x)

        return x
    torch_inter_output = torch_inter_model(torch_input)
    if tf_model.name == 'resnet34':
        import ipdb; ipdb.set_trace()
    tf_inter_model = tf.keras.Model(
        tf_model.input,
        # for resnet18, layer 1 is good, as well as 1st downsampling in layer 2
        # however 1st conv in layer 2 is not working
        # ['input_1', 'conv1_pad', 'conv1_conv', 'conv1_bn', 'conv1_relu', 'pool1_pad', 'pool1_pool',
        # 'conv2_block1_1_conv', 'conv2_block1_1_bn', 'conv2_block1_1_relu', 'conv2_block1_2_conv',
        # 'conv2_block1_2_bn', 'conv2_block1_add', 'conv2_block1_out', 'conv2_block2_1_conv',
        # 'conv2_block2_1_bn', 'conv2_block2_1_relu', 'conv2_block2_2_conv', 'conv2_block2_2_bn',
        # 'conv2_block2_add', 'conv2_block2_out',
        # 'conv3_block1_1_conv', 'conv3_block1_1_bn',
        # 'conv3_block1_1_relu', 'conv3_block1_0_conv', 'conv3_block1_2_conv', 'conv3_block1_0_bn',
        # 'conv3_block1_2_bn', 'conv3_block1_add', 'conv3_block1_out', 'conv3_block2_1_conv',
        # 'conv3_block2_1_bn', 'conv3_block2_1_relu', 'conv3_block2_2_conv', 'conv3_block2_2_bn',
        # 'conv3_block2_add', 'conv3_block2_out',
        # 'conv4_block1_1_conv', 'conv4_block1_1_bn',
        # 'conv4_block1_1_relu', 'conv4_block1_0_conv', 'conv4_block1_2_conv', 'conv4_block1_0_bn',
        # 'conv4_block1_2_bn', 'conv4_block1_add', 'conv4_block1_out', 'conv4_block2_1_conv',
        # 'conv4_block2_1_bn', 'conv4_block2_1_relu', 'conv4_block2_2_conv', 'conv4_block2_2_bn',
        # 'conv4_block2_add', 'conv4_block2_out',
        # 'conv5_block1_1_conv', 'conv5_block1_1_bn',
        # 'conv5_block1_1_relu', 'conv5_block1_0_conv', 'conv5_block1_2_conv', 'conv5_block1_0_bn',
        # 'conv5_block1_2_bn', 'conv5_block1_add', 'conv5_block1_out', 'conv5_block2_1_conv',
        # 'conv5_block2_1_bn', 'conv5_block2_1_relu', 'conv5_block2_2_conv', 'conv5_block2_2_bn',
        # 'conv5_block2_add', 'conv5_block2_out',
        # 'avg_pool', 'predictions']
        tf_model.get_layer('conv3_block1_1_conv').output if tf_model.name == 'resnet18' else tf_model.get_layer('conv2_block3_out').output,
    )
    import ipdb; ipdb.set_trace()
    tf_inter_output = tf_inter_model(tf_input, training=False)
    np_tf_output = np.transpose(tf_inter_output.numpy(), (0, 3, 1, 2))
    np_torch_output = torch_inter_output.detach().numpy()
    np.testing.assert_almost_equal(
        np_tf_output,
        np_torch_output,
        decimal=5,
    )
    print(f'done {tf_model.name}')
    # np.testing.assert_almost_equal(tf_output.numpy(), torch_output.detach().numpy())
    break
