import os
import pdb

fo = open("../data/val_Taipei.txt", "w")
path = '/media/VSlab2/BCTsai/Lab/datasets/NTHU_512/imgs/val/Taipei'
include_label = False

if include_label:
    for file in os.listdir(path):
        fo.write(os.path.join(path,file)+' '+os.path.join(path.replace('imgs','labels'),file.replace('.png','_eval.png'))+'\n')
else:
    for file in os.listdir(path):
        fo.write(os.path.join(path,file)+' '+'*\n')

fo.close()
