from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon.loss import _reshape_like, SigmoidBinaryCrossEntropyLoss, Loss
from scipy.special import expit
import timeit
from evaluation import results
from utilities import SimpleLRScheduler, get_iterators
from models import getResNet, get_fsr
from attention import attention_net_trainer
import numpy as np



def train(args, ctx):
    cl_weights = mx.nd.array(
        [1.0, 3.4595959, 18.472435, 3.3854823, 3.5971165, 1.1370194, 12.584616, 5.7822747, 10.827924, 1.7478157,
         8.8111115, 28.433332, 2.7568319, 18.020712])
    batch_ratios = nd.array(1 / cl_weights, ctx=ctx)

    class WeightedFocal(Loss):
        def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
            super(WeightedFocal, self).__init__(weight, batch_axis, **kwargs)
            self._from_sigmoid = from_sigmoid

        def hybrid_forward(self, F, pred, label, sample_weight=None):
            label = _reshape_like(F, label, pred)
            if not self._from_sigmoid:
                max_val = F.relu(-pred)
                loss = pred - pred * label + max_val + F.log(F.exp(-max_val) + F.exp(-pred - max_val))
            else:
                p = mx.nd.array(1 / (1 + nd.exp(-pred)), ctx=ctx)
                weights = nd.exp(label + (1 - label * 2) * batch_ratios)
                gamma = 2
                w_p, w_n = nd.power(1. - p, gamma), nd.power(p, gamma)
                loss = - (w_p * F.log(p + 1e-12) * label + w_n * F.log(1. - p + 1e-12) * (1. - label))
                loss *= weights
            return F.mean(loss, axis=self._batch_axis, exclude=True)


    class AttHistory(Loss):
        def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
            super(AttHistory, self).__init__(weight, batch_axis, **kwargs)
            self._from_sigmoid = from_sigmoid

        def hybrid_forward(self, F, pred, label, sample_weight=None):
            label = _reshape_like(F, label, pred)
            if not self._from_sigmoid:
                max_val = F.relu(-pred)
                loss = pred - pred * label + max_val + F.log(F.exp(-max_val) + F.exp(-pred - max_val))
            else:
                p = mx.nd.array(1 / (1 + nd.exp(-pred)), ctx=ctx)
                if epoch >= history_track and not args.test:
                    p_hist = prediction_history[:, batch_id * args.batch_size: (batch_id + 1) * args.batch_size, :]
                    p_std = (np.var(p_hist, axis=0) + (np.var(p_hist, axis=0)**2)/(p_hist.shape[0] - 1))**.5
                    std_weights = nd.array(1 + p_std, ctx=ctx)
                    loss = - std_weights * (F.log(p + 1e-12) * label + F.log(1. - p + 1e-12) * (1. - label))
                else:
                    loss = - (F.log(p + 1e-12) * label + F.log(1. - p + 1e-12) * (1. - label))
            return F.mean(loss, axis=self._batch_axis, exclude=True)


    history_track, start_history = 5, 2
    prediction_history = nd.zeros((history_track, 22968, args.num_classes), ctx=ctx)
    data_shape = (3, 224, 224)
    lr_scheduler = SimpleLRScheduler(learning_rate=args.lr)
    train_iter, val_iter, _ = get_iterators(args.batch_size, args.num_classes, data_shape)
    model = getResNet(args.num_classes, ctx, args.finetune)
    if not args.finetune:
        all_params = model.collect_params()
        trainer = gluon.Trainer(all_params, optimizer='sgd', optimizer_params={'lr_scheduler': lr_scheduler,
                                                                               'momentum': args.mom,
                                                                               'wd': args.wd})
    stages, stage_attentions, stage_trainers = {}, {}, {}
    for stage in range(2, 5):
        stride = 1
        stage_attentions['stage_' + str(stage)] = attention_net_trainer(lr_scheduler, args.num_classes, args, stride, ctx)
        kernel = 14
        if stage == 4:
            kernel = 7
        stages['stage_' + str(stage)] = get_fsr(args.num_classes, ctx, kernel_size=kernel)
        stage_trainers['stage_' + str(stage)] = gluon.Trainer(stages['stage_' + str(stage)].collect_params(),
                                                                  optimizer='sgd',
                                                                  optimizer_params={'lr_scheduler': lr_scheduler,
                                                                                    'momentum': args.mom,
                                                                                    'wd': args.wd})

    sigmoid_loss = WeightedFocal()
    attention_loss = AttHistory()
    patience, lr_drops = 0, 0

    best_mAP = 0
    smoothing_constant, moving_loss_tr, moving_loss_val = .01, 0.0, 0.0

    # Load Models
    model.load_params('saved_models/backbone_resnet.params', ctx=ctx)

    for stage in range(2, 5):
        stages['stage_' + str(stage)].load_params('saved_models/fsr_stage_' + str(stage) + '.params', ctx=ctx)
        stage_attentions['stage_' + str(stage)][0].load_params('saved_models/fconv_stage_' + str(stage) + '.params', ctx=ctx)
        stage_attentions['stage_' + str(stage)][1].load_params('saved_models/fatt_stage_' + str(stage) + '.params', ctx=ctx)

    print('Training Starts')
    for epoch in range(args.epochs):
        train_iter.reset()
        val_iter.reset()
        start_time = timeit.default_timer()

        predicts_tr, labels_tr = [], []
        for batch_id, batch in enumerate(train_iter):
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)

            with autograd.record():
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

                    temp_f = nd.reshape(output_att, (output_att.shape[0] * output_att.shape[1], output_att.shape[2] * output_att.shape[3]))
                    spatial_attention = nd.reshape(nd.softmax(temp_f), (output_att.shape[0], output_att.shape[1], output_att.shape[2], output_att.shape[3]))

                    attention_features = spatial_attention*features
                    all_stages['stage_' + str(stage)] = stages['stage_' + str(stage)](attention_features)
                    if stage == 2:
                        loss = attention_loss(all_stages['stage_' + str(stage)], label)
                    else:
                        loss = loss + attention_loss(all_stages['stage_' + str(stage)], label)

                if not args.finetune:
                    loss_original = sigmoid_loss(output, label)
                    loss = loss + loss_original
            loss.backward()

            for stage in range(2, 5):
                stage_trainers['stage_' + str(stage)].step(data.shape[0])
                stage_attentions['stage_' + str(stage)][2].step(data.shape[0])
                stage_attentions['stage_' + str(stage)][3].step(data.shape[0])

            if not args.finetune:
                trainer.step(data.shape[0])

            curr_loss = nd.mean(loss).asscalar()
            moving_loss_tr = (curr_loss if ((batch_id == 0) and (epoch == 0))
                              else (1 - smoothing_constant) * moving_loss_tr + smoothing_constant * curr_loss)


            predictions = expit(.25*(sum(all_stages.values()) + output).asnumpy())

            if epoch >= start_history:
                prediction_history[1:, batch_id * args.batch_size:(batch_id + 1) * args.batch_size] = \
                    prediction_history[0:-1, batch_id * args.batch_size:(batch_id + 1) * args.batch_size]
                prediction_history[0, batch_id * args.batch_size:(batch_id + 1) * args.batch_size] = predictions


            predicts_tr.extend(predictions)
            labels_tr.extend(label.asnumpy())

        # Validation Set
        predicts_val, labels_val = [], []

        for val_batch_id, val_batch in enumerate(val_iter):
            data = val_batch.data[0].as_in_context(ctx)
            label = val_batch.label[0].as_in_context(ctx)

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

                temp_f = nd.reshape(output_att, (output_att.shape[0] * output_att.shape[1], output_att.shape[2] * output_att.shape[3]))
                spatial_attention = nd.reshape(nd.softmax(temp_f), (output_att.shape[0], output_att.shape[1], output_att.shape[2], output_att.shape[3]))

                attention_features = spatial_attention * features
                all_stages['stage_' + str(stage)] = stages['stage_' + str(stage)](attention_features)
                if stage == 2:
                    loss = sigmoid_loss(all_stages['stage_' + str(stage)], label)
                else:
                    loss = loss + sigmoid_loss(all_stages['stage_' + str(stage)], label)

            if not args.finetune:
                loss_original = sigmoid_loss(output, label)
                loss = loss + loss_original

            curr_loss = nd.mean(loss).asscalar()
            moving_loss_val = (curr_loss if ((val_batch_id == 0) and (epoch == 0))
                               else (1 - smoothing_constant) * moving_loss_val + smoothing_constant * curr_loss)

            predictions = expit(.25 * (sum(all_stages.values()) + output).asnumpy())  # + output

            predicts_val.extend(predictions)
            labels_val.extend(label.asnumpy())

        # Evaluation
        elapsed_time = timeit.default_timer() - start_time
        train_mAP, train_APs, val_mAP, val_APs = results(labels_tr, predicts_tr, labels_val, predicts_val, epoch,
                                                         moving_loss_tr, moving_loss_val, elapsed_time)

        # if epoch == 3:
        #     patience = 0
        #     lr_scheduler.learning_rate /= float(10)
        #     print("New Learning Rate=%f" % lr_scheduler.learning_rate)


        # Model Saving
        if val_mAP > best_mAP:
            patience = 0
            best_mAP = val_mAP
            # for stage in range(2, 5):
            #     stages['stage_' + str(stage)].save_params('attention_models/joint_new_fsr_stage_' + str(stage) + '.params')
            #     stage_attentions['stage_' + str(stage)][0].save_params('attention_models/joint_new_fconv_stage_' + str(stage) + '.params')
            #     stage_attentions['stage_' + str(stage)][1].save_params('attention_models/joint_new_fatt_stage_' + str(stage) + '.params')
            # if not args.finetune:
            #     model.save_params('saved_models/joint_new_base_resNet.params')
            print('Model Saved with mAP=%f' % best_mAP)
        else:
            patience += 1
            if patience > 10 and lr_drops < 3:
                lr_drops += 1
                lr_scheduler.learning_rate /= float(10)
                print("New Learning Rate=%f" % lr_scheduler.learning_rate)
                patience = 0
