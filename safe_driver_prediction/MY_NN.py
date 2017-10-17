import torch 
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import sys

class My_Net(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes,dropout,batchnorm):
        super(My_Net,self).__init__()
        self.num_layers = len(hidden_size)+1
        layer_size = [input_size] + hidden_size+ [num_classes]
        self.nn =nn.Sequential()
        for layer in range(self.num_layers):
            idx = layer+1
            self.nn.add_module('fc'+str(idx),nn.Linear(layer_size[layer], layer_size[layer+1]))
            if layer != self.num_layers-1:
                if batchnorm is not None:
                        self.nn.add_module('batchnorm'+str(idx),
                                           nn.BatchNorm1d(num_features=layer_size[layer+1],eps=1e-8))
                self.nn.add_module('relu'+str(idx),nn.ReLU())
                if dropout is not None:
                    self.nn.add_module('dropout'+str(idx),nn.Dropout(p=dropout))
                else:
                    pass
        #batchnorm layer bias 0, and gamma seems 0.5-1? No need to initialize on purpose.
        for name, module in self.nn.named_children():
            if 'fc' in name:
                init.xavier_normal(module.weight)
                init.constant(module.bias,0.1)
                
        # for layer in range(self.num_layers):
        #     self.nn.add_module('fc'+str(layer+1), nn.Linear(layer_size[layer], layer_size[layer+1]))
        #     #self.fc['fc'+str(layer+1)] = nn.Linear(layer_size[layer], layer_size[layer+1])
        # self.relu = nn.ReLU()
        
    def forward(self, x):
        # out = self.fc['fc1'](x)
        # for i in range(self.num_layers-1):
        #     idx = i+2
        #     out = self.relu(out)
        #     out = self.fc['fc'+str(idx)](out)
        
        # Plan: B
        # module_dict= dict(self.named_modules())
        # for i in range(self.num_layers):
        #     idx = i+1
        #     if idx ==1:
        #         out = module_dict['fc'+str(idx)](x)
        #         continue
        #     out = self.relu(out)
        #     out = module_dict['fc'+str(idx)](out)
        # return out
        
        out = self.nn.forward(x)
        return out

class NeuralNetwork():
    def __init__(self,data, **kwargs):
        
        self.X_train = data['X_train']
        self.X_val = data['X_val']
        self.y_train = data['y_train']
        self.y_val = data['y_val']
                
        self.input_size = kwargs.pop('input_size',235)
        self.hidden_size = kwargs.pop('hidden_size',[150,120,80])
        self.batch_size = kwargs.pop('batch_size',1000)
        self.num_classes= kwargs.pop('num_classes',2)
        self.num_epochs = kwargs.pop('num_epochs',5)
        self.learning_rate = kwargs.pop('learning_rate',5e-4)
        
        self.lr_decay = kwargs.pop('lr_decay',
                                   {'step_size':30, 'gamma':1e-1}) #L2 reg
        self.weight_decay = kwargs.pop('weight_decay',1e-5) #L2 reg
        self.dropout = kwargs.pop('dropout', 0.5)
        self.verbose = kwargs.pop('verbose', True)
        self.batchnorm = kwargs.pop('batchnorm', True)
        self.net = My_Net(self.input_size, self.hidden_size,self.num_classes,
                          self.dropout,self.batchnorm)
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                        lr=self.learning_rate,weight_decay=self.weight_decay)
        if self.lr_decay is not None:
            self.scheduler= StepLR(self.optimizer,
                                   step_size=self.lr_decay['step_size'],gamma= self.lr_decay['gamma'])
        #print(self.net.parameters)
        if len(kwargs)>0:
            raise ValueError('Unrecognized arguments!!')


    def train(self):
        self._reset()
        X_train_tensor = torch.from_numpy(self.X_train.values)
        X_val_tensor = torch.from_numpy(self.X_val.values)
        X_train_tensor = X_train_tensor.float()
        X_val_tensor = X_val_tensor.float()
        
        y_train_tensor = torch.from_numpy(self.y_train.values)

        train_set = Data.TensorDataset(data_tensor=X_train_tensor, target_tensor=y_train_tensor)

        data_loader = Data.DataLoader(
                dataset=train_set,      
                batch_size=self.batch_size,      # mini batch size
                shuffle=True,               # random shuffle for training
                num_workers=2)              # subprocesses for loading data

        for epoch in range(self.num_epochs):
            self.scheduler.step()
            # for param_group in self.optimizer.param_groups:
            #     #print(param_group['lr']) # there is also a param called lr_init blabla here
            #     print(param_group)
            #     break
            #     sys.exit(0)               
            for i, (batch_x, batch_y) in enumerate(data_loader):
                #print(batch_x)
                self.net.train()
                x,  y = Variable(batch_x), Variable(batch_y)
                out = self.net(x)
                loss = self.criterion(out, y)
                self.optimizer.zero_grad() # clear gradients for next train
                loss.backward()
                self.optimizer.step()
                self.loss_history.append(loss.data.numpy())
                if i % 100 ==0:
                    self.net.eval()
                    acc_train = self.check_accuracy(X_train_tensor, self.y_train)
                    acc_val = self.check_accuracy(X_val_tensor, self.y_val)
                    
                    auc_train = self.check_auc(X_train_tensor, self.y_train)
                    auc_val = self.check_auc(X_val_tensor, self.y_val)
                    
                    self.acc_history['train'].append(acc_train)
                    self.acc_history['val'].append(acc_val)
                    self.auc_history['train'].append(auc_train)
                    self.auc_history['val'].append(auc_val)
                    
                    if self.verbose == True:
                        print('Epoch {}: iteration {}, the loss is {}'.format(epoch
                                                                              ,i ,loss.data.numpy()))
                        print('  acc for train: {}, acc for val: {}'.format(acc_train, acc_val))
                        print('  auc for train: {}, auc for val: {}'.format(auc_train,auc_val))
                        print('--------------------------------------------------------------')
                        
    
    def _reset(self):
        self.loss_history= []
        self.acc_history={}
        self.auc_history={}
        
        self.acc_history['train'] = []
        self.acc_history['val'] = []
        self.auc_history['train'] = []
        self.auc_history['val'] = []

                
    def check_accuracy(self,X_tensor,y):
        X_Variable = Variable(X_tensor)
        out= self.net(X_Variable)
        out = F.softmax(out)
        pred_y = np.argmax(out.data.numpy(),axis=1)
        acc = np.mean(pred_y==y.values)
        return acc
    
    def check_auc(self, X_tensor,y):
        X_Variable = Variable(X_tensor)
        out= self.net(X_Variable)
        out = F.softmax(out)
        auc = roc_auc_score(y.values, out.data.numpy()[:,1])
        return auc