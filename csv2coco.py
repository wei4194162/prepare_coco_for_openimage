import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
# from IPython import embed
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
np.random.seed(41)
# from Dataset import *
# 0为背景
from multiprocessing.pool import ThreadPool
import time
classes='/mnt/sda1/openimage2019/csv2coco/class_descriptions_boxable.csv'

classid=pd.read_csv(classes)

classlist=classid['code'].tolist()
namelist=classid['name'].tolist()
classname_to_id={}
code_to_id={}
for index,c in enumerate(namelist):
    classname_to_id[c]=index
    code_to_id[classid['code'][index]]=index
    print (c,":",classname_to_id[c],':',classid['code'][index])

# classname_to_id = {"person": 1}


class Csv2CoCo:

    def __init__(self, image_dir, total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        # pool = multiprocessing.Pool(processes=3)
        pool = ThreadPool(processes=8)

        self._init_categories()
        self.keys=keys


        # imagekeys=pool.map(self._image,keys)

        start = time.time()

        self.imagekeys=pool.map(self._image,keys)
        pool.close()
        pool.join()
        end = time.time()

        print('\nimagekey process',len(self.imagekeys),'in\n:',end-start,'per',len(self.imagekeys)/(end-start),'/sec\n\n')
        # exit(0)
        # self.image_s=self.imagekeys


        for key in tqdm(keys):

            # self.imagekey=self._image(key)
            #
            # if self.imagekey==0:
            #     continue
            # self.images.append(self.imagekey)
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi=shape[:-1]
                label = shape[-1]
                annotation = self._annotation(bboxi, label)

                # if display:
                    # print('\ndraw in ',annotation['bbox'])
                    # c=annotation['bbox']
                    # self.img = cv2.rectangle(self.img, (c[0],c[1]), (c[0]+c[2],c[1]+c[3]), (0, 0, 255), 2)
                    # self.img=cv2.putText(self.img,label,(c[0],c[1]),cv2.FONT_ITALIC,1,(255,0,0),1)
                    # cv2.imwrite(str(index)+'.jpg',self.img)

                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1

        instance = {}
        instance['info'] = 'weizhaoyu created'
        instance['license'] = ['license']
        instance['images'] = self.imagekeys
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        # print(path)
        img = cv2.imread(path)
        # img=pool.map(cv2.imread,[path])[0]

        # print(img)
        if type(img)==type(None):
            return 0
        self.img=img
        # image_file=open('path','rb')
        # my_image = Image(image_file)

        # print(path,'\nresolution :',my_image.y_resolution,my_image.x_resolution)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        self.image=image
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, label):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        # annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        #(xmax.xmin.ymax.ymin)
        # print('\npoints',points,'\n')
        min_x = int(float(points[0])*self.image['width'])
        min_y = int(float(points[2])*self.image['height'])
        max_x = int(float(points[1])*self.image['width'])
        max_y = int(float(points[3])*self.image['height'])
        # print(min_x,min_y, max_x-min_x,max_y-min_y)
        return [min_x,min_y, max_x-min_x,max_y-min_y]
        #xywh
    # segmentation
    # def _get_seg(self, points):
    #     min_x = points[0]
    #     min_y = points[2]
    #     max_x = points[1]
    #     max_y = points[3]
    #     h = max_y - min_y
    #     w = max_x - min_x
    #     a = []
    #     a.append([min_x, min_y, min_x, min_y + 0.5 * h, min_x, max_y, min_x + 0.5 * w, max_y, max_x, max_y, max_x,
    #               max_y - 0.5 * h, max_x, min_y, max_x - 0.5 * w, min_y])
    #     return a



if __name__ == '__main__':
    csv_file = "/mnt/sda1/openimage2019/csv2coco/validation-annotations-bbox.csv"
    image_dir = "/mnt/sda1/openimage2019/dataset/validation"
    saved_coco_path = "/mnt/sda1/openimage2019/csv2coco/data/"
    # 整合csv格式标注文件
    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file).values

    for annotation in tqdm(annotations):
        # print (annotation)
        # from exif import Image
        key = annotation[0].split(os.sep)[-1]
        key=os.path.join(image_dir,key+'.jpg')
        #XMin,XMax,YMin,YMax
        value = np.array([[annotation[4]
                            ,annotation[5]
                            ,annotation[6]
                            ,annotation[7]
                            ,namelist[code_to_id[annotation[2]]]]])
        # if total_csv_annotations == {}:
        #     print('is none')
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
        else:
            total_csv_annotations[key] = value
    # print(total_csv_annotations)
    # # 按照键值划分数据
    total_keys = list(total_csv_annotations.keys())
    # print(total_keys[3])
    # print(total_csv_annotations[total_keys[3]])
    train_keys, val_keys = train_test_split(total_keys, test_size=0.000000001)
    print("train_n:", len(train_keys), 'val_n:', len(val_keys))
    # 创建必须的文件夹
    if not os.path.exists('%scoco/annotations/' % saved_coco_path):
        os.makedirs('%scoco/annotations/' % saved_coco_path)
    if not os.path.exists('%scoco/images/train2017/' % saved_coco_path):
        os.makedirs('%scoco/images/train2017/' % saved_coco_path)
    if not os.path.exists('%scoco/images/val2017/' % saved_coco_path):
        os.makedirs('%scoco/images/val2017/' % saved_coco_path)
    # 把训练集转化为COCO的json格式
    l2c_train = Csv2CoCo(image_dir=image_dir, total_annos=total_csv_annotations)
    train_instance = l2c_train.to_coco(train_keys)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)

    for file in train_keys:
        shutil.copy(file, "%scoco/images/train2017/" % saved_coco_path)
    for file in val_keys:
        shutil.copy(file, "%scoco/images/val2017/" % saved_coco_path)


    # 把验证集转化为COCO的json格式
    l2c_val = Csv2CoCo(image_dir=image_dir, total_annos=total_csv_annotations)
    val_instance = l2c_val.to_coco(val_keys)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)
