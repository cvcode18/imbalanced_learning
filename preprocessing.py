from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import misc
from utils import get_data
def resize_images():

    full_path = ADD YOUR DATA PATH HERE

    image_path = full_path
    annotation_path = full_path + 'Annotations/'
    size = (256, 256)

    all_im_list_tr = np.array([line.rstrip('\n')[1:-2] for line in open(full_path+'wider_att/wider_att_train_imglist.txt')])
    im_list_test = np.array([line.rstrip('\n')[1:-2] for line in open(full_path+'wider_att/wider_att_test_imglist.txt')])

    # Saves Images to the same folder. Make a copy of the initial folder first
    for im in all_im_list_tr:
        img = Image.open(image_path + im[1:])
        img_res = img.resize(size, Image.ANTIALIAS)
        img_res.save(full_path[:-1] + im, 'JPEG')

    for im in im_list_test:
        img = Image.open(image_path + im[1:])
        img_res = img.resize(size, Image.ANTIALIAS)
        img_res.save(full_path[:-1] + im, 'JPEG')


def save2lists(im_list, att_list, filename):
    L = []
    for c, im in enumerate(im_list):
        tmp = list(att_list[c])
        L.append([str(c)]+map(str,tmp)+[str(im)])
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(L)

def data_prep(full_path):
    im_list_tr, att_list_tr, im_list_val, att_list_val, im_list_test, att_list_test = get_data(full_path)
    save2lists(im_list_tr, att_list_tr,'training_list.lst')
    save2lists(im_list_val, att_list_val,'valid_list.lst')
    save2lists(im_list_test, att_list_test,'testing_list.lst')
