#getdata.py
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


IMAGE_H= 200
IMAGE_W= 200


dataTransform = transforms.Compose([
    transforms.ToTensor()   # Tensor형식을 전환，수치귀일화[0.0, 1.0]
])

class DogsVSCatsDataset(data.Dataset):      
    def __init__(self, mode, dir):         
        self.mode = mode
        self.list_img = []                  
        self.list_label = []                
        self.data_size = 0                 
        self.transform = dataTransform      

        if self.mode == 'train':            
            dir = dir + '/train/'          
            for file in os.listdir(dir):   
                self.list_img.append(dir + file)        
                self.data_size += 1                     
                name = file.split(sep='.')           
                # label가 one-hot를 적용，
                if name[0] == 'cat':
                    self.list_label.append(0)        
                else:
                    self.list_label.append(1)         
        elif self.mode == 'test':        
            dir = dir + '/test/'           
            for file in os.listdir(dir):
                self.list_img.append(dir + file)    
                self.data_size += 1
                self.list_label.append(2)      
        else:
            print('Undefined Dataset!')

    def __getitem__(self, item):            
        if self.mode == 'train':                                     
            img = Image.open(self.list_img[item])                      
            img = img.resize((IMAGE_H, IMAGE_W))                             
            img = np.array(img)[:, :, :3]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])     
        elif self.mode == 'test':                                     
            img = Image.open(self.list_img[item])
            img = img.resize((IMAGE_H, IMAGE_W))
            img = np.array(img)[:, :, :3]
            return self.transform(img)         
        else:
            print('None')

    def __len__(self):
        return self.data_size              

