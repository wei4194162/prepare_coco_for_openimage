# 一、准备dataset
把自己的数据集用csv2coco.py转化成coco数据集格式。
<h4 id="1">1. 内容介绍</h4>

介绍如何将csv格式数据及转化为coco格式：

- csv to coco


<h4 id="2">2. 标准格式</h4>

在使用转换脚本之前，必须要明确的几种格式

<h5 id="2.1">2.1 csv</h5>

不要一看是`csv`文件就直接拿来运行，如果不是，可以自行修改代码，或者修改标注文件。

转换脚本支持的csv格式应为以下形式:

- `csv/`
    - `labels.csv`
    - `images/`
        - `image1.jpg`
        - `image2.jpg`
        - `...`

`labels.csv` 的形式: 

`/path/to/image,xmin,ymin,xmax,ymax,label`

例如:

```
/mfs/dataset/face/0d4c5e4f-fc3c-4d5a-906c-105.jpg,450,154,754,341,dog
/mfs/dataset/face/0ddfc5aea-fcdac-421-92dad-144.jpg,143,154,344,341,cat
...
```
注：图片路径请使用绝对路径


<h5 id="2.3">2.2 coco</h5>

此处未使用测试集

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



<h4 id="3">3. 如何使用转换脚本</h4>

<h5 id="3.1">3.1 csv2coco</h5>

首先更改`csv2coco.py`中以下几个配置

```
classname_to_id = {"person": 1}  # for your dataset classes
csv_file = "labels.csv"  # annatations file path
image_dir = "images/"    # original image path
saved_coco_path = "./"   # path to save converted coco dataset
```

然后运行 `python csv2coco.py`

会自动创建文件夹并复制图片到相应位置，运行结束后得到如下：

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

<h5 id="3.2">3.2 csv2voc</h5>

首先更改`csv2voc.py`中以下几个配置

```
csv_file = "labels.csv"
saved_path = ".VOC2007/" # path to save converted voc dataset     
image_save_path = "./JPEGImages/"   # converted voc images path
image_raw_parh = "images/"          # original image path
```

然后运行 `python csv2voc.py`

# 二、Training前修改相关文件
首先说明的是我的数据集类别一共有601个。使用哪个模型修改对应模型的配置文件，这里以该模型为例：’configs/faster_rcnn_r50_fpn_1x.py’。
<h4 id="1">1. 定义数据种类</h4>
需要修改的地方在mmdetection/mmdet/datasets/coco.py。把CLASSES的那个tuple改为自己数据集对应的种类tuple即可。例如：
```
CLASSES = ('dog', 'cat')
```
<h4 id="1">2. 修改coco_classes数据集类别</h4>
需要修改的地方在mmdetection/mmdet/core/evaluation/class_names.py。
```
def coco_classes():
    return [
        'dog', 'cat'
    ]
```
<h4 id="1">3. 修改对应模型配置文件</h4>
修改configs/faster_rcnn_r50_fpn_1x.py中的model字典中的num_classes、data字典中的img_scale和optimizer中的lr(学习率)。例如：
```
num_classes=3,#类别数+1
img_scale=(640,478), #输入图像尺寸的最大边与最小边（train、val、test这三处都要修改）
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001) #当gpu数量为8时,lr=0.02；当gpu数量为4时,lr=0.01；我只要一个gpu，所以设置lr=0.0025
```
```
num_classes=3,#类别数+1
img_scale=(640,478), #输入图像尺寸的最大边与最小边（train、val、test这三处都要修改）
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001) #当gpu数量为8时,lr=0.02；当gpu数量为4时,lr=0.01；我只要一个gpu，所以设置lr=0.0025
```
