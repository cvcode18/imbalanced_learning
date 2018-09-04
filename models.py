from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
import mxnet as mx
from mxnet import gluon


def get_fsr(num_classes, ctx, kernel_size):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=256, kernel_size=1))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(channels=512, kernel_size=1))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(channels=1024, kernel_size=kernel_size))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Dense(num_classes, flatten=True))
    net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)

    return net


def get_fatt(num_classes, stride, ctx):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=512, kernel_size=1))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        # net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1))
        # net.add(nn.BatchNorm())
        # net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(channels=num_classes, kernel_size=1, strides=stride))
    net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
    return net


def get_conv2D(num_classes, stride, ctx):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=num_classes, kernel_size=1, strides=stride))
        net.add(nn.Activation('sigmoid'))
    net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
    return net


def getResNet(num_classes, ctx, NoTraining=True):
    resnet = vision.resnet101_v1(pretrained=True, ctx=ctx)

    net = vision.resnet101_v1(classes=num_classes, prefix='resnetv10_')
    with net.name_scope():
        net.output = nn.Dense(num_classes, flatten=True, in_units=resnet.output._in_units)
        net.output.collect_params().initialize(
            mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
        net.features = resnet.features

    net.collect_params().reset_ctx(ctx)

    inputs = mx.sym.var('data')
    out = net(inputs)
    internals = out.get_internals()
    outputs = [internals['resnetv10_stage3_activation19_output'], internals['resnetv10_stage3_activation22_output'], internals['resnetv10_stage4_activation2_output'],
               internals['resnetv10_dense1_fwd_output']]
    feat_model = gluon.SymbolBlock(outputs, inputs, params=net.collect_params())
    feat_model._prefix = 'resnetv10_'
    if NoTraining:
        feat_model.collect_params().setattr('grad_req', 'null')
    return feat_model



def getDenseNet(num_classes, ctx):
    densenet = vision.densenet201(pretrained=True, ctx=ctx)

    net = vision.densenet201(classes=num_classes, prefix='densenet0_')
    with net.name_scope():
        net.output = nn.Dense(num_classes, flatten=True)
        net.output.collect_params().initialize(
            mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
        net.features = densenet.features

    net.collect_params().reset_ctx(ctx)

    inputs = mx.sym.var('data')
    out = net(inputs)
    internals = out.get_internals()
    outputs = [internals['densenet0_conv3_fwd_output'], internals['densenet0_stage4_concat15_output'],
               internals['densenet0_dense1_fwd_output']]
    feat_model = gluon.SymbolBlock(outputs, inputs, params=net.collect_params())
    feat_model._prefix = 'densenet0_'

    return feat_model
