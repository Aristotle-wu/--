from optparse import Values
import os
import csv
import json
from typing_extensions import dataclass_transform

data_path = "Attention-based-Skin-Cancer-Classification-main/Ham10000 models/archive/HAM10000"
data_dict = {}
data_class = os.listdir(data_path)
use_labels = []
category_id = []

for c in data_class:
    data_class_path = os.path.join(data_path, c)
    class_labels = os.listdir(data_class_path)

    for i in range(len(class_labels)):
        class_path = os.path.join(data_class_path, class_labels[i])
        class_data = os.listdir(class_path)

        for f in class_data:
            data_dict[f] = class_labels[i]
            category_id.append(i + 1)
            if c == "test_dir":
                use_labels.append("test")
            else:
                use_labels.append("train")

keys = list(data_dict.keys())
values = list(data_dict.values())

train_images = []
train_categories = []
train_annotations = []
val_images = []
val_categories = []
val_annotations = []

train_file = open("instances_train2017.json", "w", encoding="utf-8")
val_file = open("instances_val2017.json", "w", encoding="utf-8")

for k in range(len(keys)):
    if use_labels[k] == "train":

        image = {"license": None, 
                "file_name": keys[k], 
                "id": k, "width": 600.0, 
                "height": 450.0}
        train_images.append(image)
        category = {"supercategory": values[k], 
                    "id": category_id[k], 
                    "name": values[k]}
        train_categories.append(category)
        annotation = {"segmentation": None,
                     "image_id": k, 
                     "category_id": category_id[k]}
        train_annotations.append(annotation)

    elif use_labels[k] == "test":

        image = {"license": None, "file_name": keys[k], "id": k, "width": 600.0, "height": 450.0}
        val_images.append(image)
        category = {"supercategory": values[k], "id": category_id[k], "name": values[k]}
        val_categories.append(category)
        annotation = {"segmentation": None, "image_id": k, "category_id": category_id[k]}
        val_annotations.append(annotation)
train_dic = {
    "info": None, 
    "licenses": None, 
    "images": train_images, 
    "annotation": train_annotations, 
    "categories": train_categories}
train_file.write(json.dumps(train_dic))
train_file.close()

test_dic = {"info": None, "licenses": None, "images": val_images, "annotation": val_annotations, "catergories": val_categories}
val_file.write(json.dumps(test_dic))
val_file.close()

print("ok")
