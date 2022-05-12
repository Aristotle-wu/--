# from pycocotools.coco import COCO
import json
import os, cv2, shutil
from unicodedata import category, name
from lxml import etree, objectify
from numpy import positive
from tqdm import tqdm
from PIL import Image

# 生成图片保存的路径
CKimg_dir = 'mmclassification-master/data/coco2017_voc/images'
# 生成标注文件保存的路径
CKanno_dir = 'mmclassification-master/data/coco2017_voc/annotations'


# 若模型保存文件夹不存在，创建模型保存文件夹，若存在，删除重建
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

# 转换成voc数据集的Annotations
def trans_Annotations(source_file_path, target_file_path):
    with open(source_file_path) as f:
        dic= json.load(f)
        images = dic['images']
        categories = dic['categories']
        for i in images:
            E = objectify.ElementMaker(annotate=False)
            j = 0
            Annotations_tree = E.annotation(
            E.folder('HAM10000'),
            E.filename(i['file_name']),
            E.source(
                E.database('HAM10000'),
                E.annotation('HAM10000_VOC2017'),
                E.image('')
            ),
            E.size(
                E.width(i['width']),
                E.height(i['height']),
                E.depth(3)
            ),
            E.segmented(0),
            E.object(
                E.name(categories[j]['supercategory']),
                E.pose(),
                E.truncated("0"),
                E.difficult(0)
                )
            )
            j += 1
            name = i['file_name']
            final_path = os.path.join(target_file_path, '{}xml'.format(name[:-3]))
            etree.ElementTree(Annotations_tree).write(final_path, pretty_print=True)

# 转换成voc数据集的ImageSets文件夹
def trans_ImageSets(source_file_path, target_file_path):
    with open(source_file_path) as f:
        dic= json.load(f)
        images = dic['images']
        if source_file_path[-12:-9] == 'val':
            target_file = os.path.join(target_file_path, 'val.txt')
            if os.path.exists(target_file):
                os.remove(target_file)
            with open(target_file, 'w') as file:
                for i in images:
                    name = i['file_name']
                    file.write('Annotations/'+ name[:-4]+'\n')

        elif source_file_path[-14:-9] == 'train':
            target_file = os.path.join(target_file_path, 'train.txt')
            if os.path.exists(target_file):
                os.remove(target_file)
            with open(target_file,'w') as file:
                for i in images:
                    name = i['file_name']
                    file.write('Annotations/'+ name[: -4]+'\n')
        file.close()
    f.close()

# 转换成voc数据集的JPEGImages文件夹
def trans_JPEGImages(source_file_train_path , source_file_val_path, target_file_path):
    train_Images = os.listdir(source_file_train_path)
    val_images = os.listdir(source_file_val_path)
    for i in train_Images:
        source = os.path.join(source_file_train_path, i)
        shutil.copy(source, target_file_path)
    for j in val_images:
        source = os.path.join(source_file_val_path, j)
        shutil.copy(source, target_file_path)
    sum_images = os.listdir(target_file_path)
    return len(sum_images)

def main():
    base_dir = 'mmclassification-master/data/HAM10000_voc2017'  # step1 这里是一个新的文件夹，存放转换后的图片和标注
    mkr(base_dir)
    JPEGImages_dir = os.path.join(base_dir, 'JPEGImages')  # 在上述文件夹中生成images，annotations两个子文件夹
    anno_dir = os.path.join(base_dir, 'Annotations')
    image_sets_dir = os.path.join(base_dir, 'ImageSets')
    image_sets_dir_Main = os.path.join(image_sets_dir, 'Main')
    mkr(JPEGImages_dir)
    mkr(anno_dir)
    mkr(image_sets_dir)
    mkr(image_sets_dir_Main)
    
    train_ann_path = 'mmclassification-master/data/HAM10000_coco/annotations/instances_train2017.json'
    val_ann_path  = 'mmclassification-master/data/HAM10000_coco/annotations/instances_val2017.json'

    source_images_train_path = 'mmclassification-master/data/HAM10000_coco/train2017'
    source_images_val_path = 'mmclassification-master/data/HAM10000_coco/val2017'

    # 转换annotations文件夹
    trans_Annotations(train_ann_path , anno_dir)
    trans_Annotations(val_ann_path , anno_dir)

    # 转换ImageSets文件夹
    trans_ImageSets(train_ann_path, image_sets_dir_Main)
    trans_ImageSets(val_ann_path, image_sets_dir_Main)

    # 转换图片文件
    sum_images = trans_JPEGImages(source_images_train_path, source_images_val_path, JPEGImages_dir)

    print(sum_images)
    
if __name__ == "__main__":
    main()
    print('ok')

