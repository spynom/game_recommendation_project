import torch
from torch import nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(35)


class GMF(nn.Module):
    def __init__(self,items_size,users_size,embedding_dim):
        super(GMF, self).__init__()
        self.items_embedding = nn.Embedding(items_size,embedding_dim)
        self.users_embedding = nn.Embedding(users_size,embedding_dim)

    def forward(self,items,users):
        item_vector = self.items_embedding(items)
        user_vector = self.users_embedding(users)
        mul_vector=user_vector*item_vector
        return torch.sigmoid(torch.sum(mul_vector,dim=1))


GMF_model=GMF(227,666536,50).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(GMF_model.parameters(), lr=0.001)

class_0=np.load("data/train_class_0.npy")[:,[0,1,-1]]
class_0=torch.from_numpy(class_0).type(torch.int32)
class_1=np.load("data/train_class_1.npy")[:,[0,1,-1]]
class_1=torch.from_numpy(class_1).type(torch.int32)

val_class_0=np.load("data/val_class_0.npy")[:,[0,1,-1]]
val_class_1=np.load("data/val_class_1.npy")[:,[0,1,-1]]
val=torch.from_numpy(np.concatenate([val_class_0,val_class_1],axis=0)).type(torch.int32).to(device)

epoch_size=20
batch_size=10000
no_of_batches=class_1.shape[0]//batch_size
class_0_no_of_batches=class_0.shape[0]//batch_size

report={"train_loss":[],"val_loss":[]}

for epoch in range(epoch_size):
    j=0
    class_0_index=np.arange(class_0.shape[0])
    class_1_index=np.arange(class_1.shape[0])
    for i in range(no_of_batches):
        GMF_model.train()
        class_0_batch_index=class_0_index[j*batch_size:(j+1)*batch_size]
        class_1_batch_index=class_1_index[i*batch_size:(i+1)*batch_size]
        x1=torch.concatenate((
            class_0[class_0_batch_index,0],
            class_1[class_1_batch_index,0],
        ),0).to(device)
        x2=torch.concatenate((
        class_0[class_0_batch_index, 1],
        class_1[class_1_batch_index, 1])
        ,0).to(device)

        label1=class_0[class_0_batch_index,-1]
        label2=class_1[class_1_batch_index,-1]
        labels=torch.concatenate((label1,label2),0).to(device).to(torch.float32)
        labels=labels

        optimizer.zero_grad()
        output=GMF_model(x1,x2)
        train_loss=criterion(output,labels)

        train_loss.backward()
        optimizer.step()

        if not i%10:
            GMF_model.eval()
            x1=val[:,0]
            x2=val[:,1]
            labels=val[:,-1].to(torch.float32)
            labels=labels
            with torch.no_grad():
                output=GMF_model(x1,x2)
                val_loss=criterion(output,labels)
            print(f'step:{i}/{no_of_batches},epoch{epoch+1}, Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
        if j%class_0_no_of_batches:
            j+=1
        else:
            np.random.shuffle(class_0_index)
            j=0
    np.random.shuffle(class_1_index)


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



torch.save(GMF_model.state_dict(), 'models/GMF_model.pth')
with open('reports/GMF_train_report.json', 'w') as f:
    json.dump(report,f)



