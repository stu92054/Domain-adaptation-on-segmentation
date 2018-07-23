import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import pdb


class SynthiaDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in self.img_ids:
            #
            img_file = osp.join(self.root, "RGB/%s" % name)
            label_file = osp.join(self.root, "GT/trainid/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
            
            #
        """
        datafiles = self.files[3056]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        #label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        label_copy = label
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        """

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        #
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        #for k, v in self.id_to_trainid.items():
            #label_copy[label == k] = v
        #
        label_copy = label

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name

if __name__ == '__main__':
    dst = SynthiaDataSet("../../../../addhd5/nitahaha/RAND_CITYSCAPES/","./Synthia_list/train.txt")
    trainloader = data.DataLoader(dst, batch_size=4)
    0
    for i, data in enumerate(trainloader):
        
        imgs, labels, size, names = data
        #pdb.set_trace()
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
