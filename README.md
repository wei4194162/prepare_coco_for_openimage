# prepare dataset
Convert openimage dataset to coco dataset format using `csv2coco.py`.
<h4 id="1">1. Introduction</h4>

Describe how to convert csv format data into coco format:

- csv to coco


<h4 id="2">2. standard format</h4>

Several formats must be specified before using the conversion script.

<h5 id="2.1">2.1 csv</h5>


The csv format supported by the conversion script should be of the form:

- `csv/`
    - `labels.csv`
    - `images/`
        - `image1.jpg`
        - `image2.jpg`
        - `...`

`labels.csv` : 

`/path/to/image,xmin,ymin,xmax,ymax,label`

for instance:

```
/mfs/dataset/face/0d4c5e4f-fc3c-4d5a-906c-105.jpg,450,154,754,341,dog
/mfs/dataset/face/0ddfc5aea-fcdac-421-92dad-144.jpg,143,154,344,341,cat
...
```

Note: Please use absolute path for image path


<h5 id="2.2">2.2 coco</h5>

coco format

- `coco/`
    - `annotations/`
        - `instances_train2017.json`
        - `instances_val2017.json`
    - `images/`
        - `train2017/`
            - `0d4c5e4f-fc3c-4d5a-906c-105.jpg`
            - `...`
        - `val2017`
            - `0ddfc5aea-fcdac-421-92dad-144.jpg`
            - `...`



<h4 id="3">3. How to use the conversion script</h4>

<h5 id="3.1">3.1 csv2coco</h5>

First change the following configuration in `csv2coco.py`

```
classname_to_id = {"person": 1}  # for your dataset classes
csv_file = "labels.csv"  # annatations file path
image_dir = "images/"    # original image path
saved_coco_path = "./"   # path to save converted coco dataset
```

Then run `python csv2coco.py`
The folder will be created automatically and the image will be copied to the corresponding location. After running the script, you will get the following:

- `coco/`
    - `annotations/`
        - `instances_train2017.json`
        - `instances_val2017.json`
    - `images/`
        - `train2017/`
            - `0d4c5e4f-fc3c-4d5a-906c-105.jpg`
            - `...`
        - `val2017`
            - `0ddfc5aea-fcdac-421-92dad-144.jpg`
            - `...`

# modify related files before Training
The first thing to note is that there are 601 categories in openimage dataset. Then modify the configuration file, here is an example of this model: `configs/faster_rcnn_r50_fpn_1x.py` .

<h4 id="1">1. Define category</h4>

In `mmdetection/mmdet/datasets/coco.py`. Change the tuple of CLASSES to the tuple of the dataset. E.g:

```
CLASSES = ('dog', 'cat')
```
Note: Use `read_classes.py` to get all the classes of openimage datasets.

<h4 id="1">2. modify coco_classes</h4>
in `mmdetection/mmdet/core/evaluation/class_names.py` .

```
def coco_classes():
    return [
        'dog', 'cat'
    ]
```

<h4 id="1">3. modify the configuration file</h4>
Modify num_classes img_scale and lr in `configs/faster_rcnn_r50_fpn_1x.py` . E.g:

```
num_classes=3,#categories+1
img_scale=(640,478), #maximum and minimum edges of the input image size (train, val, test all three places must be modified)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001) #When the number of gpu is 8, lr=0.02; when the number of gpu is 4, lr=0.01
```
