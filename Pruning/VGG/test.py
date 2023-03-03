import os

DATASET_DIR= "/home/cguo/SCTest/Pruning/VGG/imagenet/"
#DATASET_DIR= "/home/fcxue/pycode/TileSparsity-master/Pruning/VGG"
#print(os.listdir(DATASET_DIR))
with open('/home/cguo/SCTest/Pruning/VGG/imagenet/validation-00118-of-00128', 'r') as f:
    data = f.read()
    print(data[0])