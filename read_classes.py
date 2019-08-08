import pandas as pd

classes='/mnt/sda1/openimage2019/csv2coco/class_descriptions_boxable.csv'

classid=pd.read_csv(classes)

classlist=classid['code'].tolist()
namelist=classid['name'].tolist()
print(namelist)
