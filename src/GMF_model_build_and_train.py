# import libraries
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json

# check cuda availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(35) # set randomness for reproducibility

# GMF architecture
class GMF(nn.Module):
    def __init__(self,items_size,users_size,embedding_dim):
        super(GMF, self).__init__()

        self.items_embedding = nn.Embedding(items_size,embedding_dim)  # defining items embedding layer

        self.users_embedding = nn.Embedding(users_size,embedding_dim)  # defining users embedding layer

    def forward(self,items,users):
        item_vector = self.items_embedding(items.reshape(-1,1)).squeeze(1) # item vector the output og embedding layer

        user_vector = self.users_embedding(users.reshape(-1,1)).squeeze(1) # user vector the output og embedding layer

        mul_vector=user_vector*item_vector # multiplying vectors element wise
        return torch.sigmoid(torch.sum(mul_vector,dim=1)) # applying sigmoid


GMF_model=GMF(227,666536,50).to(device) # initialize model
criterion = nn.BCELoss() # setting up loss function
optimizer = optim.Adam(GMF_model.parameters(), lr=0.001) # setting optimizer

class_0=np.load("../data/train_class_0.npy")[:,[0,1,-1]] # loading class 0 contain train file
class_0=torch.from_numpy(class_0).type(torch.int32) # converting into torch tensor
class_1=np.load("../data/train_class_1.npy")[:,[0,1,-1]] # loading class 1 contain train file
class_1=torch.from_numpy(class_1).type(torch.int32) # converting into torch tensor

val_class_0=np.load("../data/val_class_0.npy")[:,[0,1,-1]] # loading class 0 contain validation file
val_class_1=np.load("../data/val_class_1.npy")[:,[0,1,-1]]  # loading class 1 contain validation file
val=torch.from_numpy(np.concatenate([val_class_0,val_class_1],axis=0)).type(torch.int32).to(device) # merging and converting into tensor

epoch_size=20 # size of epoch
batch_size=10000 # size of batch
no_of_batches=class_1.shape[0]//batch_size # no of batches
class_0_no_of_batches=class_0.shape[0]//batch_size # separate no of batches for class 0 to handel imbalance

report={"train_loss":[],"val_loss":[]} # dic to save report of training

for epoch in range(epoch_size):
    j=0 # reading the batch no. for class 0
    class_0_index=np.arange(class_0.shape[0]) # indexes of class 0
    class_1_index=np.arange(class_1.shape[0]) # indexes of class 1
    for i in range(no_of_batches):
        GMF_model.train() # change mode of model to train
        class_0_batch_index=class_0_index[j*batch_size:(j+1)*batch_size]
        class_1_batch_index=class_1_index[i*batch_size:(i+1)*batch_size]
        x1=torch.concatenate((
            class_0[class_0_batch_index,0],
            class_1[class_1_batch_index,0],
        ),0).to(device) # game id attribute
        x2=torch.concatenate((
        class_0[class_0_batch_index, 1],
        class_1[class_1_batch_index, 1])
        ,0).to(device) # user id attribute

        label1=class_0[class_0_batch_index,-1]
        label2=class_1[class_1_batch_index,-1]
        labels=torch.concatenate((label1,label2),0).to(device).to(torch.float32)
        labels=labels # target attribute

        optimizer.zero_grad() # reset gradient of optimizer
        output=GMF_model(x1,x2) # y_hat
        train_loss=criterion(output,labels) # calculate loss

        train_loss.backward() # applying backward propagation
        optimizer.step() # update parameters

        if not i%10:
            GMF_model.eval() # change model mode evaluation
            x1=val[:,0]
            x2=val[:,1]
            labels=val[:,-1].to(torch.float32)
            labels=labels
            with torch.no_grad():
                output=GMF_model(x1,x2) # val y_hat
                val_loss=criterion(output,labels) # val loss
            print(f'step:{i}/{no_of_batches},epoch{epoch+1}, Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
        if j%class_0_no_of_batches:
            j+=1 # updating j batch no of class 0
        else:
            np.random.shuffle(class_0_index) # add randomness in index of class 0 instances
            j=0 # reset batch no of class 0
    np.random.shuffle(class_1_index) # add randomness in index of class 1 instances

    # evaluating model performance on overall train dataset
    x1=torch.concatenate((
        class_0[:,0],
        class_1[:,0],
    ),0).to(device)
    x2=torch.concatenate((
        class_0[:,1],
        class_1[:,1],
    ),0).to(device)


    label1=class_0[:,-1]
    label2=class_1[:,-1]
    labels=torch.concatenate((label1,label2),0).to(torch.float32).to(device)

    GMF_model.eval()
    with torch.no_grad():
        output=GMF_model(x1,x2)
        train_loss=criterion(output,labels)
        report["train_loss"].append(train_loss.item())
        report["val_loss"].append(val_loss.item())



torch.save(GMF_model.state_dict(), '../models/GMF_model.pth') # saving state of model
with open('../reports/GMF_train_report.json', 'w') as f:
    json.dump(report,f) # saving training report as json file



