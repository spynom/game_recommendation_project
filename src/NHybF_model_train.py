import torch
from torch import nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
# check cuda availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set randomness for reproducibility
np.random.seed(34)
torch.manual_seed(57)

# architecture of NHybF
class NHybF(nn.Module):
    def __init__(self, num_items,num_users, items_embedding_dim,users_embedding_dim,gmf_embedding_dim,hidden_layers_size=(50,50,50,50),dropout_prob=(0.25,0.25,0.25,0.25),output_size=1):
        super(NHybF,self).__init__()


        # HybMLP architecture part
        self.item_embedding = nn.Embedding(num_embeddings=num_items,embedding_dim=items_embedding_dim)# items embedding layer

        self.user_embedding = nn.Embedding(num_embeddings=num_users,embedding_dim=users_embedding_dim) # items embedding layer

        # hidden layer 1
        self.hidden_layer1 = nn.Linear(in_features=items_embedding_dim+users_embedding_dim+358, out_features=hidden_layers_size[0])
        self.dropout1 = nn.Dropout(dropout_prob[0])
        # hidden layer 2
        self.hidden_layer2=nn.Linear(in_features=hidden_layers_size[0],out_features=hidden_layers_size[1])
        self.dropout2 = nn.Dropout(dropout_prob[1])
        # hidden layer 3
        self.hidden_layer3=nn.Linear(in_features=hidden_layers_size[1],out_features=hidden_layers_size[2])
        self.dropout3 = nn.Dropout(dropout_prob[2])
        # hidden layer 4
        self.hidden_layer4=nn.Linear(in_features=hidden_layers_size[2],out_features=hidden_layers_size[3])
        self.dropout4 = nn.Dropout(dropout_prob[3])
        # output layer of HyMLP
        self.HybMLP_output_layer = nn.Linear(in_features=hidden_layers_size[3],out_features=output_size)


        # GMF architecture part
        self.items_embedding1 = nn.Embedding(num_items,gmf_embedding_dim)
        self.users_embedding1 = nn.Embedding(num_users,gmf_embedding_dim)
        self.gmf_dropout = nn.Dropout(dropout_prob[4])

        self.output_layer=nn.Linear(in_features=output_size+gmf_embedding_dim,out_features=1)

    def forward(self,x1,x2,x3):
        item_vector=self.item_embedding(x1)
        user_vector=self.user_embedding(x2)
        concatenation=torch.cat((item_vector, user_vector, x3), dim=1)

        first_hidden=self.hidden_layer1(concatenation)
        dropout_output1=self.dropout1(first_hidden)
        relu1=F.relu(dropout_output1)

        second_hidden=self.hidden_layer2(relu1)
        dropout_output2=self.dropout2(second_hidden)
        relu2=F.relu(dropout_output2)

        third_hidden=self.hidden_layer3(relu2)
        dropout_output3=self.dropout3(third_hidden)
        relu3=F.relu(dropout_output3)

        forth_hidden=self.hidden_layer4(relu3)
        dropout_output4=self.dropout4(forth_hidden)
        relu4=F.relu(dropout_output4)

        HybMLP_output=self.HybMLP_output_layer(relu4)


        GMF_item_vector = self.items_embedding1(x1)
        GMF_user_vector = self.users_embedding1(x2)
        GMF_mul_vector=GMF_item_vector*GMF_user_vector
        GMF_output=self.gmf_dropout(GMF_mul_vector)

        concatenation=torch.cat((HybMLP_output, GMF_output), dim=1)

        output=self.output_layer(concatenation)

        return torch.sigmoid(output)

NHybF_model=NHybF(num_items=227,num_users=666536,items_embedding_dim= 20,users_embedding_dim=40,gmf_embedding_dim=20,output_size=15,hidden_layers_size=(128,16,128,32),dropout_prob=(0.75,0.5,0.75,0.5,0.5)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(NHybF_model.parameters(), lr=0.001)


class_0=np.load("data/train_class_0.npy")
class_0=torch.from_numpy(class_0).type(torch.float32)
class_1=np.load("data/train_class_1.npy")
class_1=torch.from_numpy(class_1).type(torch.float32)

val_class_0=np.load("data/val_class_0.npy")
val_class_1=np.load("data/val_class_1.npy")
val=torch.from_numpy(np.concatenate([val_class_0,val_class_1],axis=0)).type(torch.float32).to(device)

epoch_size=8
batch_size=10000
no_of_batches=class_1.shape[0]//batch_size
class_0_no_of_batches=class_0.shape[0]//batch_size


report={"train_loss":[],"val_loss":[]}
for epoch in range(epoch_size):
    NHybF_model.to(device)
    j=0
    class_0_index=np.arange(class_0.shape[0])
    class_1_index=np.arange(class_1.shape[0])
    for i in range(no_of_batches):
        NHybF_model.train()
        class_0_batch_index=class_0_index[j*batch_size:(j+1)*batch_size]
        class_1_batch_index=class_1_index[i*batch_size:(i+1)*batch_size]
        x1=torch.concatenate((
            class_0[class_0_batch_index,0],
            class_1[class_1_batch_index,0],
        ),0).to(device).to(torch.int32)
        x2=torch.concatenate((
            class_0[class_0_batch_index,1],
            class_1[class_1_batch_index,1],
        ),0).to(device).to(torch.int32)
        x3=torch.concatenate((
            class_0[class_0_batch_index,2:-1],
            class_1[class_1_batch_index,2:-1],
        ),0).to(device)

        label1=class_0[class_0_batch_index,-1]
        label2=class_1[class_1_batch_index,-1]
        labels=torch.concatenate((label1,label2),0).to(device)
        labels=labels.reshape(-1,1)
        optimizer.zero_grad()
        output=NHybF_model(x1,x2,x3)
        train_loss=criterion(output,labels)
        train_loss.backward()
        optimizer.step()
        if not i%10:
            NHybF_model.eval()
            x1=val[:,0].to(torch.int32)
            x2=val[:,1].to(torch.int32)
            x3=val[:,2:-1]
            labels=val[:,-1]
            labels=labels.reshape(-1,1)
            with torch.no_grad():
                output=NHybF_model(x1,x2,x3)
                val_loss=criterion(output,labels)
            print(f'step:{i}/{no_of_batches},epoch{epoch+1}, Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
        if j%class_0_no_of_batches:
            j+=1
        else:
            np.random.shuffle(class_0_index)
            j=0

        if val_loss<0.525 and train_loss<0.525 and epoch>6:
            break

    np.random.shuffle(class_1_index)


    x1=torch.concatenate((
        class_0[:,0],
        class_1[:,0],
    ),0).to(torch.int32)
    x2=torch.concatenate((
        class_0[:,1],
        class_1[:,1],
    ),0).to(torch.int32)
    x3=torch.concatenate((
        class_0[:,2:-1],
        class_1[:,2:-1],
    ),0)

    label1=class_0[:,-1]
    label2=class_1[:,-1]
    labels=torch.concatenate((label1,label2),0).reshape(-1,1)

    NHybF_model.to("cpu").eval()
    with torch.no_grad():
        output=NHybF_model(x1,x2,x3)
        train_loss=criterion(output,labels)
        report["train_loss"].append(train_loss.item())
        report["val_loss"].append(val_loss.item())


torch.save(NHybF_model.state_dict(), 'models/NHybF_model.pth')

with open('reports/NHybF_train_report.json', 'w') as f:
    json.dump(report,f)