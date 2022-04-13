from keras.applications.resnet import ResNet
from keras import layers
import numpy as np
from torchsummary import summary
from torchvision import models

PRINT_SUMMARY = False



torch_resnet18 = models.resnet18(pretrained=True)
torch_resnet34 = models.resnet34(pretrained=True)


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

tf_resnet18 = ResNet18(weights=None, use_bias=True, input_shape=(224, 224, 3))
tf_resnet34 = ResNet34(weights=None, use_bias=True, input_shape=(224, 224, 3))

for torch_model, tf_model in zip(
    [torch_resnet18, torch_resnet34],
    [tf_resnet18, tf_resnet34],
):
    if PRINT_SUMMARY:
        summary(torch_model, input_size=(3, 224, 224))
        tf_model.summary()
    torch_weights_map = {
        name: param
        for name, param in torch_model.named_parameters()
    }
    torch_buffers_map = {
        name: param
        for name, param in torch_model.named_buffers()
    }
    torch_weights_map.update(**torch_buffers_map)

    def apply_conv_torch_weights_to_tf(tf_layer, torch_layer_name):
        tf_weights = tf_layer.get_weights()
        torch_weights = torch_weights_map[f'{torch_layer_name}.weight']
        bias = tf_weights[1]
        reshaped_torch_weights = np.transpose(torch_weights.detach().numpy(), (2, 3, 1, 0))
        tf_layer.set_weights([reshaped_torch_weights, bias])

    def apply_bn_torch_weights_to_tf(tf_layer, torch_layer_name):
        torch_bias = torch_weights_map[f'{torch_layer_name}.bias']
        torch_scale = torch_weights_map[f'{torch_layer_name}.weight']
        torch_mean = torch_weights_map[f'{torch_layer_name}.running_mean']
        torch_var = torch_weights_map[f'{torch_layer_name}.running_var']
        # order for tf bn weights is
        # ['conv1_bn/gamma:0', 'conv1_bn/beta:0', 'conv1_bn/moving_mean:0', 'conv1_bn/moving_variance:0']
        # and according to the docs, gamma is the scale and beta is the bias
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
        tf_layer.set_weights([
            torch_scale.detach().numpy(),
            torch_bias.detach().numpy(),
            torch_mean.detach().numpy(),
            torch_var.detach().numpy(),
        ])

    for layer in tf_model.layers:
        weights = layer.get_weights()
        if weights:
            layer_ids = layer.name.split('_')
            layer_type = layer_ids[-1]
            if 'block' in layer.name:
                i_layer = str(int(layer_ids[0][4:]) -1)
                i_block = str(int(layer_ids[1][5:]) - 1)
                i_conv = layer_ids[2]
                torch_layer_name = f'layer{i_layer}.{i_block}'
                if layer_type == 'conv':
                    if int(i_conv) > 0:
                        torch_layer_name += f'.conv{i_conv}'
                    else:
                        # downsampling layer
                        torch_layer_name += '.downsample.0'
                    apply_conv_torch_weights_to_tf(layer, torch_layer_name)
                elif layer_type == 'bn':
                    if int(i_conv) > 0:
                        torch_layer_name += f'.bn{i_conv}'
                    else:
                        # downsampling layer
                        torch_layer_name += '.downsample.1'
                    apply_bn_torch_weights_to_tf(layer, torch_layer_name)
                else:
                    raise ValueError(f'Unknown layer type: {layer_type}')
            else:
                if layer_type == 'conv':
                    apply_conv_torch_weights_to_tf(layer, 'conv1')
                elif layer_type == 'bn':
                    apply_bn_torch_weights_to_tf(layer, 'bn1')
                elif layer_type == 'predictions':
                    torch_weights = torch_weights_map['fc.weight']
                    reshaped_torch_weights = np.transpose(torch_weights.detach().numpy(), (1, 0))
                    torch_bias = torch_weights_map['fc.bias']
                    layer.set_weights([reshaped_torch_weights, torch_bias.detach().numpy()])
                else:
                    raise ValueError(f'Unknown layer type: {layer_type} for 1st conv')
    tf_model.save_weights(f'{tf_model.name}.h5')
