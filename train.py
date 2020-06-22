# train.py
from getdata import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn

dataset_dir = './data/test/'                    # dataset경로          
model_cp = './model/model.pth'           #모델의위치            
workers = 10                                        #PyTorch가 DataLoader의수 읽기
batch_size = 20                                     #batch_size크기
lr = 0.0001                                            #학습된확률


def train():
       datafile = DVCD('train', dataset_dir)
       dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers)
       
        print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

       model = Net()                                                    #네트워크실제화
       model = model.cuda()                                        # GPU로 계산
       model.train()                                                      # trainnig모델
  
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)    #adam
      
       criterion = torch.nn.CrossEntropyLoss()   #CrossEntropyLoss 함수

       cnt = 0
  
      for img, label in dataloader:
           img, label = Variable(img).cuda(), Variable(label).cuda()
           out = model(img)                                    
           loss = criterion(out, label.squeeze())
           loss.backward() 
           optimizer.step()                                         
           optimizer.zero_grad()                                      
           cnt += 1                     
           print('Frame {0}, train_loss {1}'.format(cnt*batch_size, loss/batch_size))   

      torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))      

if __name__ == '__main__':
    train()

         