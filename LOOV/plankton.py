import torch
import torch.nn as nn
import torchnet
import sparseconvnet as scn
import cv2
import sparseconvnet as scn
import pickle
import math
import random
import numpy as np
import os
import glob


dataDir='zooscan_group1'
classes=sorted([x.split('/')[1] for x in glob.glob(dataDir+'/*')])

# two-dimensional SparseConvNet
m=2
class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.SparseVggNet(2, 1, [
            ['C', m*8 ], ['C', m*8 ], 'MP',
            ['C', m*16], ['C', m*16], 'MP',
            ['C', m*24], ['C', m*24], 'MP',
            ['C', m*32], ['C', m*32], 'MP',
            ['C', m*40], ['C', m*40]]
            ).add(scn.AveragePooling(2,100,1)
            ).add(scn.SparseToDense(2,m*40))
        self.bn = nn.BatchNorm1d(m*40)
        self.linear = nn.Linear(m*40, len(classes))
    def forward(self, x):
        x = self.sparseModel(x)
        x = x.view(-1,m*40)
        x = self.bn(x)
        x = self.linear(x)
        return x
precomputeStride=3
model=Model()
print(model)
spatial_size = model.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
print('Input spatial size:', spatial_size)

def dataset(directory, classes, spatial_size, train, low=0, high=1.01):
    d=[{'input': input, 'target': i} for i, cl in enumerate(classes) for input in sorted(glob.glob(directory+'/'+cl+'/*.*'))]
    np.random.seed(0)
    filter=[low<=u<high for u in np.random.uniform(0,1,len(d))]
    d=[x for x,f in zip(d,filter) if f]
    for idx,x in enumerate(d):
        x['idx']=idx
    print(directory,len(d))
    d = torchnet.dataset.ListDataset(d)

    randperm = torch.randperm(len(d))
    def perm(idx, size):
        return randperm[idx]

    def merge(tbl):
        inp = scn.InputBatch(2, spatial_size)
        np_random = np.random.RandomState(tbl['idx'])
        for sample in tbl['input']:
            img=255-cv2.imread(sample,cv2.IMREAD_GRAYSCALE)
            src = np.float32([[0           ,           0],
                              [img.shape[1],           0],
                              [0           ,img.shape[0]],
                              [img.shape[1],img.shape[0]]])
            m = np.float32([[1,0],[0,1]])
            if train:
                m+=np_random.uniform(-0.1,0.1,(2,2))
            theta=np_random.uniform(0,2*math.pi)
            m=np.dot(m,np.float32([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]]))
            dst = np.dot(src, m)
            dst-=dst.min(0)
            sz=dst.max(0)
            b=dst.max(0)-dst.min(0)
            m = cv2.getAffineTransform(src[:3],dst[:3])
            img = cv2.warpAffine(img,m,tuple(sz))
            img=img[:,:,None]
            tensor=torch.from_numpy(np.array(img,dtype='float32'))/255
            offset = spatial_size/2+torch.LongTensor(2).random_(-16,15)
            inp.addSampleFromTensor(tensor, offset, threshold=0)
        inp.precomputeMetadata(precomputeStride)
        return {'input': inp, 'target': torch.LongTensor(tbl['target']), 'idx': torch.LongTensor(tbl['idx'])}
    bd = torchnet.dataset.BatchDataset(d, 100, perm=perm, merge=merge)
    tdi = scn.threadDatasetIterator(bd)
    def iter():
        randperm.copy_(torch.randperm(len(d)))
        return tdi()
    return iter


dataset = {'train': dataset(dataDir, classes, spatial_size, True, 0, 0.9),
           'val': dataset(dataDir, classes, spatial_size, False, 0.99, 1.1)}

scn.ClassificationTrainValidate(
    model, dataset,
    {'n_epochs': 10,
    'initial_lr': 0.1,
    'lr_decay': 0.5,
    'weight_decay': 1e-4,
    'use_gpu':  torch.cuda.is_available(),
    'check_point': True,
    'test_reps': 3})
