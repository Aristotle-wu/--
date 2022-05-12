import os
import shutil
from numpy import source
val_path = 'mmclassification-master/data/HAM10000/test_dir'
train_path = 'mmclassification-master/data/HAM10000/train_dir'

val_tag = os.listdir(val_path)
train_tag = os.listdir(train_path)

# os.mkdir('mmclassification-master/data/HAM10000/train2017')
# os.mkdir('mmclassification-master/data/HAM10000/val2017')

for i in val_tag:
    path = os.path.join(val_path, i)
    image_path = os.listdir(path)
    for img in image_path:
        source_path = os.path.join(path, img)
        target_path = os.path.join('mmclassification-master/data/HAM10000/val2017/', img)
        shutil.copyfile(source_path, target_path)

for i in train_tag:
    path = os.path.join(train_path, i)
    image_path = os.listdir(path)
    for img in image_path:
        source_path = os.path.join(path, img)
        target_path = os.path.join('mmclassification-master/data/HAM10000/train2017/', img)
        shutil.copyfile(source_path, target_path)

len_train= len(os.listdir('mmclassification-master/data/HAM10000/train2017/'))
len_test = len(os.listdir('mmclassification-master/data/HAM10000/val2017/'))

print('ok')

    
