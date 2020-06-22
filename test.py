from getdata import DogsVSCatsDataset as DVCD
from network import Net
import torch
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

dataset_dir = './data/test/'                  # dataset경로
model_file = './model/model.pth'                #모델경로

 def test():

     model = Net()                                        #네트워크실제화
     model.cuda()                                         # GPU로 계산
     model.load_state_dict(torch.load(model_file))       # 학습된 모델 로딩
     model.eval()                                        # eval모델

     datafile = DVCD('test', dataset_dir)                # dataset 실례화
     print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

     index = np.random.randint(0, datafile.data_size, 1)[0]      # random수로 image를 임의 선택함
     img = datafile.__getitem__(index)                           # image 하다 받다
     img = img.unsqueeze(0)                                      
     img = Variable(img).cuda()                                  # Data를 PyTorch의 Variable노드에서 놓고 ,또는 GPU에서 들어가고 시작점을 담임힘.
     out = model(img)                                            
     out = F.softmax(out, dim=1)                                        # SoftMax으로 2个개 output값을 [0.0, 1.0]을 시키다,합이1이다.
     print(out)                      # output는 개/고양이의 확률
     if out[0, 0] > out[0, 1]:                   # 개<고양이
         print('the image is a cat')
     else:                                       # 개>고양이
         print('the image is a dog')

     img = Image.open(datafile.list_img[index])      # text image open
     plt.figure('image')                             # matplotlib로  image show
     plt.imshow(img)
     plt.show()

if __name__ == '__main__':
    test()


