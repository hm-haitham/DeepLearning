import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config
import torch.optim as optim


def one_hot_encoding(input_, nb_classes): 
    tmp = input_.new_zeros(input_.size()[0], nb_classes)  #tensor of zeros (batch_size,10)
    tmp.scatter_(1, input_, 1)     #fill with one in the correct index
    return tmp

def train_siamese(model,dataloader, epochs, learning_rate, aux_loss = False, weight_loss_1 = 0.4, weight_loss_2 = 0.4):
    
    model.train() #set the model on training mode 
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
      
    print("Start training with {0} epochs, a learning rate of {1} and {2} as loss function".format(epochs,learning_rate, criterion))
        
    if(aux_loss):
        print("With auxilary loss function")
    else: 
        print("Without auxilary loss function")
            
    if(model.nb_subnet == 1):
        print("With weight sharing")
    else:
        print("Without weight sharing")
           
    training_losses = []
    training_acc = []
    
    for epoch in range(1, epochs+1):
        
        sum_loss_epoch = 0
        total = 0
        correct = 0
        accuracy_epoch = 0
        
        for ind_batch, sample_batched in enumerate(dataloader):
    
            images = sample_batched["images"]                                  #(batch_size,2,14,14)
            compare_labels = sample_batched["bool_labels"].float().view(-1,1)  #(batch_size,1)
            digit_labels = sample_batched["digit_labels"]                      #(batch_size,2)
            
            
            #gets (batch_size,1) and returns (batch_size,10)
            one_hot_encoded_label1 = one_hot_encoding(digit_labels[:,0].view(-1,1), nb_classes=10)
            one_hot_encoded_label2 = one_hot_encoding(digit_labels[:,1].view(-1,1), nb_classes=10)
            
            loss = 0
            optimizer.zero_grad()
        
            #seperate the 2 batches of images
            input1 = images[:,0:1,:,:]   #(batch_size,1,14,14)
            input2 = images[:,1:2,:,:]   #(batch_size,1,14,14)
        
            #compute the forward pass
            output1, output2, output = model.forward(input1,input2)
            
            if(aux_loss):   
                #Cross entropy loss but as already applying the soft_max activation function
                #so only need to apply log
                #add small value to avoid the log(0) problem
                #multiplication element wise to keep only the probability of the correct label
            
                loss_input1 = -torch.log(output1 + 1e-20) * one_hot_encoded_label1.float()   #(batch_size,1)
                loss_input2 = -torch.log(output2 + 1e-20) * one_hot_encoded_label2.float()   #(batch_size,1)
            
                loss = weight_loss_1 * loss_input1.mean() + weight_loss_2 * loss_input2.mean()  #auxilary loss
           
            loss += criterion(output, compare_labels)   #if batched do the mean of the errors 

            loss.require_grad = True   #should remove I think
            loss.backward()

            optimizer.step()
            
            #update the accuracy 
            total += images.size(0)  
            correct += (output.round() == compare_labels).sum()  
            
            if ind_batch % 250 == 0:
                print("[Epoch {}, Batch {}/{}]:  [Loss: {:.2f}]".format(epoch, ind_batch, len(dataloader), loss) )
                
            #add the loss for this batch to the total loss of the epoch
            sum_loss_epoch = sum_loss_epoch + loss 
            
        #compute the mean to obtain the loss for this epoch 
        mean_loss = sum_loss_epoch / float(len(dataloader))
        
        print("At epoch {0} the loss is {1}".format(epoch, mean_loss) )
        training_losses.append(mean_loss)
        
        accuracy_epoch = float(correct) / float(total)
        print("At epoch {0} the accuracy is {1}".format(epoch, accuracy_epoch) )
        training_acc.append(accuracy_epoch)
        
    return training_losses, training_acc

def test_siamese(model, dataloader,aux_loss = False, weight_loss_1 = 0.4, weight_loss_2 = 0.4):
    
    model.eval() # set the model on evaluation mode

    criterion = nn.BCELoss()
    
    sum_test_loss = 0
    total = 0
    correct = 0
    
    for ind_batch, sample_batched in enumerate(dataloader):
        
        images = sample_batched["images"]                                  #(batch_size,2,14,14)
        compare_labels = sample_batched["bool_labels"].float().view(-1,1)  #(batch_size,1)
        digit_labels = sample_batched["digit_labels"]                      #(batch_size,2)    
            
        #gets (batch_size,1) and returns (batch_size,10)
        one_hot_encoded_label1 = one_hot_encoding(digit_labels[:,0].view(-1,1), nb_classes=10)
        one_hot_encoded_label2 = one_hot_encoding(digit_labels[:,1].view(-1,1), nb_classes=10)        
        
        #seperate the 2 batches of images
        input1 = images[:,0:1,:,:]   #(batch_size,1,14,14)
        input2 = images[:,1:2,:,:]   #(batch_size,1,14,14)
        
        #compute the forward pass
        output1, output2, output = model.forward(input1,input2)
        
        if(aux_loss):
            loss_input1 = -torch.log(output1 + 1e-20) * one_hot_encoded_label1.float()   #(batch_size,1)
            loss_input2 = -torch.log(output2 + 1e-20) * one_hot_encoded_label2.float()   #(batch_size,1)
            sum_test_loss += weight_loss_1 * loss_input1.mean() + weight_loss_2 * loss_input2.mean()  #auxilary loss
  
        sum_test_loss += criterion(output, compare_labels) 
         
        total += images.size(0)  
        correct += (output.round() == compare_labels).sum() 
    
    test_loss = sum_test_loss / float(len(dataloader))
    test_acc = float(correct) / float(total)
    
    return test_loss.item(),test_acc