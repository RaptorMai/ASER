from utils.resnet_utils import _conv, _fc, _bn, _residual_block, _residual_block_first
import utils.utils as utils
from copy import deepcopy
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def fully_model(model):
    hidden_size = deepcopy(model.args.embed_dim)
    hidden_size.append(model.out_dim)
    hidden_size.insert(0, model.in_dim[0])
    no_layers = len(hidden_size) - 1
    W = []
    b = []
    for i in range(no_layers):
        din = hidden_size[i]
        dout = hidden_size[i + 1]
        with tf.variable_scope('layer' + str(i)):
            Wi = utils.weight_variable([din, dout])
            bi = utils.bias_variable([dout])
        model.trainable_vars.append(Wi)
        model.trainable_vars.append(bi)
        W.append(Wi)
        b.append(bi)

    act = model.x
    with tf.variable_scope('hidden'):
        for i in range(no_layers - 1):
            pre = tf.add(tf.matmul(act, W[i]), b[i])
            act = tf.nn.relu(pre)
        model.features = act
    with tf.variable_scope('prediction'):
        model.y = tf.add(tf.matmul(act, W[-1]), b[-1])




def resnet18(model, h, kernels, filters, strides):
    model.trainable_vars = []

    # Conv1
    h = _conv(h, kernels[0], filters[0], strides[0], model.trainable_vars, name='conv_1')
    h = _bn(h, model.trainable_vars, model.train_phase, name='bn_1')
    h = tf.nn.relu(h)

    # Conv2_x
    h = _residual_block(h, model.trainable_vars, model.train_phase, name='conv2_1')
    h = _residual_block(h, model.trainable_vars, model.train_phase, name='conv2_2')

    # Conv3_x
    h = _residual_block_first(h, filters[2], strides[2], model.trainable_vars, model.train_phase, name='conv3_1',
                              is_ATT_DATASET=None, pretrained=model.args.pretrained)
    h = _residual_block(h, model.trainable_vars, model.train_phase, name='conv3_2')

    # Conv4_x
    h = _residual_block_first(h, filters[3], strides[3], model.trainable_vars, model.train_phase, name='conv4_1',
                              is_ATT_DATASET=None, pretrained=model.args.pretrained)
    h = _residual_block(h, model.trainable_vars, model.train_phase, name='conv4_2')

    # Conv5_x
    h = _residual_block_first(h, filters[4], strides[4], model.trainable_vars, model.train_phase, name='conv5_1',
                              is_ATT_DATASET=None, pretrained=model.args.pretrained)
    h = _residual_block(h, model.trainable_vars, model.train_phase, name='conv5_2')

    # Apply average pooling
    h = tf.reduce_mean(h, [1, 2])

    # Store the feature mappings
    model.features = h
    model.image_feature_dim = h.get_shape().as_list()[-1]

    if model.class_attr is not None:
        # Return the image features
        return h
    else:
        if model.arch == 'resnet18_s':
            logits = _fc(h, model.out_dim, model.trainable_vars, name='fc_1', is_cifar=True)
        else:
            logits = _fc(h, model.out_dim, model.trainable_vars, name='fc_1')

    with tf.variable_scope('prediction'):
        model.y = logits

def resnet_params(name):
    if name == 'resnet18_s':
        # Same resnet-18 as used in GEM paper
        kernels = [3, 3, 3, 3, 3]
        filters = [20, 20, 40, 80, 160]
        strides = [1, 0, 2, 2, 2]
    else:
        # Standard ResNet-18
        kernels = [7, 3, 3, 3, 3]
        filters = [64, 64, 128, 256, 512]
        strides = [2, 0, 2, 2, 2]
    return kernels, filters, strides
