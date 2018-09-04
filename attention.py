import numpy as np
from mxnet import nd
from models import get_conv2D, get_fatt
from mxnet import gluon
def get_action_labels(path_list):
    action_list = []
    for p in path_list:
        action_list.append(p.split("/")[3].split("--")[1])
    d = {key: value for (value, key) in enumerate(set(action_list))}

    actions = []
    for action in action_list:
        actions.append(d[action])
    return np.array(actions)


def compute_attention(features, fconv, fatt):
    output_conv = fconv(features)
    output_att = fatt(features)
    temp_f = nd.reshape(output_att,
                        (output_att.shape[0] * output_att.shape[1], output_att.shape[2] * output_att.shape[3]))
    spatial_softmax = nd.reshape(nd.softmax(temp_f),
                                 (output_att.shape[0], output_att.shape[1], output_att.shape[2], output_att.shape[3]))
    return output_conv, spatial_softmax

def attention_net_trainer(lr_scheduler, classes, args, stride, ctx):
    fconv_stg = get_conv2D(classes, stride, ctx)
    fatt_stg = get_fatt(classes, stride, ctx)

    trainer_conv, trainer_att = [], []
    if not args.test:
        trainer_conv = gluon.Trainer(fconv_stg.collect_params(), optimizer='sgd',
                                     optimizer_params={'lr_scheduler': lr_scheduler,
                                                       'momentum': args.mom,
                                                       'wd': args.wd})

        trainer_att = gluon.Trainer(fatt_stg.collect_params(), optimizer='sgd',
                                    optimizer_params={'lr_scheduler': lr_scheduler,
                                                      'momentum': args.mom,
                                                      'wd': args.wd})

    return fconv_stg, fatt_stg, trainer_conv, trainer_att


def attention_cl(lr_scheduler, args, ctx, kernel_size = 14):
    fsr_stg = get_fsr(args.num_classes, ctx, kernel_size)
    trainer_sr = gluon.Trainer(fsr_stg.collect_params(), optimizer='sgd',
                               optimizer_params={'lr_scheduler': lr_scheduler,
                                                 'momentum': args.mom,
                                                 'wd': args.wd})
    return fsr_stg, trainer_sr
