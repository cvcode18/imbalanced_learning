from models import getResNet, get_fsr
from mxnet import nd
from utilities import get_iterators
from attention import attention_net_trainer
from evaluation import evaluate_mAP
from utilities import prettyfloat
import numpy as np
from scipy.special import expit


def test(args, ctx):
    print('Testing Performance')
    data_shape = (3, 224, 224)
    _, _, test_iter = get_iterators(args.batch_size, args.num_classes, data_shape)
    test_iter.reset()
    model = getResNet(args.num_classes, ctx)
    model.load_params('saved_models/backbone_resnet.params', ctx=ctx)

    stage_attentions, stages = {}, {}
    for stage in range(2, 5):
        stage_attentions['stage_' + str(stage)] = attention_net_trainer(_, args.num_classes, args, 1, ctx)
        stage_attentions['stage_' + str(stage)][0].load_params('saved_models/fconv_stage_' + str(stage) + '.params', ctx=ctx)
        stage_attentions['stage_' + str(stage)][1].load_params('saved_models/fatt_stage_' + str(stage) + '.params', ctx=ctx)
        kernel = 14
        if stage == 4:  # 4
            kernel = 7
        stages['stage_' + str(stage)] = get_fsr(args.num_classes, ctx, kernel_size=kernel)
        stages['stage_' + str(stage)].load_params('saved_models/fsr_stage_' + str(stage) + '.params', ctx=ctx)

    predicts_test, labels_test = [], []
    for batch_id, batch in enumerate(test_iter):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        net_features_stg3_v1, net_features_stg3, net_features_stg4, output = model(data)
        all_stages = {}
        for stage in range(2, 5):
            if stage == 2:
                inp_feats = net_features_stg3_v1
            elif stage == 3:
                inp_feats = net_features_stg3
            else:
                inp_feats = net_features_stg4

            features = stage_attentions['stage_' + str(stage)][0](inp_feats)
            output_att = stage_attentions['stage_' + str(stage)][1](inp_feats)

            temp_f = nd.reshape(output_att, (
                output_att.shape[0] * output_att.shape[1], output_att.shape[2] * output_att.shape[3]))
            spatial_attention = nd.reshape(nd.softmax(temp_f), (
                output_att.shape[0], output_att.shape[1], output_att.shape[2], output_att.shape[3]))

            attention_features = spatial_attention * features
            all_stages['stage_' + str(stage)] = stages['stage_' + str(stage)](attention_features)
        predictions = expit(.25 * (sum(all_stages.values()) + output).asnumpy())

        predicts_test.extend(predictions)
        labels_test.extend(label.asnumpy())

    test_mAP, test_APs = evaluate_mAP(np.array(labels_test), np.array(predicts_test), testingFlag=True)
    print(test_mAP, list(map(prettyfloat, test_APs)))