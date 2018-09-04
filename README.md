This repository re-implements the ECCV 2018 paper [Deep Imbalanced Attribute Classification using Visual Attention Aggregation](https://arxiv.org/abs/1807.03903) 

If you use this code, please mention this repo and cite the paper:
```
@InProceedings{Sarafianos_2018_ECCV,
author = {Sarafianos, Nikolaos and Xu, Xiang and Kakadiaris, Ioannis A.},
title = {Deep Imbalanced Attribute Classification using Visual Attention Aggregation},
booktitle = {ECCV},
year = {2018}
}
```
# Development Environment

* Python 3.5

* MXNet with CUDA-9
```
$ pip install --upgrade mxnet-cu90
```
* Add project path to ```PYTHONPATH```
```
$ export PYTHONPATH=/project/path:$PYTHONPATH
$ cd /project/path
```

# Download Datasets

* WIDER-Attribute: The original images and the annotation files are provided [here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html), cropped images for each human bounding box can be downloaded [here](https://github.com/zhufengx/SRN_multilabel). 28,340 cropped images in "train" and "val" for training, 29,177 cropped images in "test" for testing.

* PETA: The original images and the annotation files are provided [here](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html). The train/val/test splits as well as the class ratio of the selected 35 attributes we used were obtained can be downloaded [here](https://github.com/asc-kit/vespa/tree/master/generated). 

# Prepare Data

In both datasets all records, list and txt files are provided in `records/`

## PETA

* Place the PETA dataset under the path `/dataset/path/PETA/PETA_dataset/` and copy paste the folder while renaming it to `/dataset/path/PETA/PETA_preproc/`. 

* Call the `resize_images` function from `preprocessing/` to resize all images to 256x256 and save them. 

* Then using the train/val/text files call `preprocessing/` which will create the .lst files for each set and save them. 

* From the initial MXNet download you should be able to find in the `tools/` the `im2rec.py` [file](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py). Open a terminal and type:

```
$ cd /incubator-mxnet/tools/
$ python im2rec.py /project/path/peta_att /dataset/path/PETA/ --quality=100 --pack-label=True
```

This will create the record files to feed to the iterator. 

## WIDER-Attribute

* Place the WIDER-Attribute dataset under the path `/dataset/path/WIDER/`. Then copy paste the images and rename as before to `Image_cropped/`. A similar approach is required in here with in which the images are resized using the function in `preprocessing/`.

* Place the downloaded annotation text files under `/dataset/path/WIDER/wider_att/`.

* Call the `data_prep` function from `preprocessing/` to obtain the image and annotation files and save them to .lst files. 

* Simlarly with above run:

```
$ cd /incubator-mxnet/tools/
$ python im2rec.py /project/path/DeepVisualAttributes /dataset/path/WIDER --quality=100 --pack-label=True
```

This will creat the record files to `wider_records/` to feed to the iterator.

## Run the Code

* For the WIDER dataset go the respective folder and run `main.py`. 

* Remember to provide as an input argumenet the data path. 
