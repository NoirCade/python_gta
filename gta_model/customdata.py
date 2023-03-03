import cv2
import torch
import os
import glob
from torch.utils.data import Dataset


# class my_dataset(Dataset) :
#     def __init__(self, path, transform=None):
#         self.path_all = glob.glob(os.path.join(path, "*", "*.png"))
#         self.transform = transform
#         self.label_dict ={"ani" : 0,
#                           "hum" : 1,
#                           }
#
#     def __getitem__(self, item):
#         # 1. image path [] -> img_path
#         img_path = self.path_all[item]
#
#         # 2. get label
#         folder_name = img_path.split("\\")[1]
#         label = self.label_dict[folder_name]
#
#         # 3. get image
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # 4. Augment an image
#         if self.transform is not None :
#             image = self.transform(image=image)["image"]
#
#         # 5. return in image, label
#         return image, label
#
#     def __len__(self):
#         return len(self.path_all)

# if __name__ == "__main__":
#     test = my_dataset("./dataset/train/", transform=None)
#     for i in test :
#         print(i)

import os
import glob
import cv2
from torch.utils.data import Dataset
from torchvision import datasets


def get_classes(data_dir) :
    all_data = datasets.ImageFolder(data_dir)
    return  all_data.classes


test = get_classes("./dataset/train/")
label_dict = {}
for i, (labels) in enumerate(test) :
    label_dict[labels] = int(i)


class my_customdata(Dataset) :
    def __init__(self, path, transform=None):
        self.all_path = glob.glob(os.path.join(path, "*", "*.jpg"))
        # [./dataset/train/ADONIS/01.jpg, ..., ..., ]
        self.transform = transform

    def __getitem__(self, item):
        image_path = self.all_path[item]
        # image read
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label
        label_tmep = image_path.split("\\")[1]
        label = label_dict[label_tmep]
        # transform
        if self.transform is not None :
            image = self.transform(image=image)["image"]

        return image, label

    def __len__(self):
        return len(self.all_path)