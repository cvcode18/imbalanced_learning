import numpy as np
from mxnet import lr_scheduler
from mxnet.io import ImageRecordIter


class prettyfloat(float):
    def __repr__(self):
        return "%0.3f" % self


class SimpleLRScheduler(lr_scheduler.LRScheduler):
    def __init__(self, learning_rate=0.1):
        super(SimpleLRScheduler, self).__init__()
        self.learning_rate = learning_rate
    def __call__(self, num_update):
        return self.learning_rate


def get_data(full_path):
    all_im_list_tr = np.array(
        [line.rstrip('\n')[1:-2] for line in open(full_path + 'wider_att/wider_att_train_imglist.txt')])
    all_att_list_tr = np.array(
        [map(int, line.rstrip(' \n').split(" ")) for line in open(full_path + 'wider_att/wider_att_train_label.txt')])

    # Split To Train and Validation
    im_list_tr = []
    att_list_tr = []
    im_list_val = []
    att_list_val = []
    for im_path, att in zip(all_im_list_tr, all_att_list_tr):
        if im_path.split("/")[2] == 'val':
            im_list_val.append(im_path)
            att_list_val.append(att)
        else:
            im_list_tr.append(im_path)
            att_list_tr.append(att)
    im_list_test = np.array([line.rstrip('\n')[1:-2] for line in open(full_path + 'wider_att/wider_att_test_imglist.txt')])
    att_list_test = np.array([map(int, line.rstrip(' \n').split(" ")) for line in open(full_path + 'wider_att/wider_att_test_label.txt')])
    return np.array(im_list_tr), np.array(att_list_tr), np.array(im_list_val), np.array(att_list_val), np.array(im_list_test), np.array(att_list_test)



def get_classweights(att_list_tr):
    class_weights = np.ones((att_list_tr.shape[1],))
    for att in xrange(att_list_tr.shape[1]):
        current_att = att_list_tr[:, att]
        cl_imb_tr = np.sum(current_att) / float(current_att.shape[0])
        if cl_imb_tr <= 0.5:
            class_weights[att] = (1 - cl_imb_tr) / float(cl_imb_tr)
    return class_weights.astype("float32")


def get_iterators(batch_size, num_classes, data_shape):
    train = ImageRecordIter(
        path_imgrec='wider_records/training_list.rec',
        path_imglist='wider_records/training_list.lst',
        batch_size=batch_size,
        data_shape=data_shape,
        preprocess_threads=4,
        mean_r=104,
        mean_g=117,
        mean_b=123,
        resize=256,
        max_crop_size=224,
        min_crop_size=128,
        label_width=num_classes,
        shuffle=False,
        round_batch=False,
        rand_crop=True,
        rand_mirror=True)
    val = ImageRecordIter(
        path_imgrec='wider_records/valid_list.rec',
        path_imglist='wider_records/valid_list.lst',
        shuffle=False,
        mean_r=104,
        mean_g=117,
        mean_b=123,
        round_batch=False,
        label_width=num_classes,
        preprocess_threads=4,
        batch_size=batch_size,
        data_shape=data_shape)
    test = ImageRecordIter(
        path_imgrec='wider_records/testing_list.rec',
        path_imglist='wider_records/testing_list.lst',
        shuffle=False,
        round_batch=False,
        mean_r=104,
        mean_g=117,
        mean_b=123,
        label_width=num_classes,
        preprocess_threads=4,
        batch_size=batch_size,
        data_shape=data_shape)
    return train, val, test
