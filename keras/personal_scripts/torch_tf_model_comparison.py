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

    if stride > 1:
        x = layers.ZeroPadding2D(padding=((1,0),(1,0)), name=name + '_1_pad')(x)
        padding_mode = 'valid'
    else:
        padding_mode = 'same'
    x = layers.Conv2D(
        filters, kernel_size, padding=padding_mode, strides=stride,
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
    np.testing.assert_almost_equal(
        tf_output.numpy(),
        torch_output.detach().numpy(),
        decimal=5,
    )
