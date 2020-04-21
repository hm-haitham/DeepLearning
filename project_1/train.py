import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from helpers import save_model

from config import TRAIN_CHECKPOINTS_DIR
from config import LEARNING_RATE

ALPHA=0.5
BETA=0.25
GAMMA=0.25

def train_double(model,dataloader,epochs,criterion,ave_model_flag=False):
    
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("CUDA available")
    else:
        print("NO CUDA")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, epochs+1):
        
        model.train()
        
        for ind_batch, sample_batched in enumerate(dataloader):
            
            images = sample_batched["images"]
            labels = sample_batched["bool_labels"]
            digit_labels = sample_batched["digit_labels"]
            
            if cuda:
                images = images.to(device="cuda")
                labels = labels.to(device="cuda")

            optimizer.zero_grad()

            output, loss_left, loss_right = model(images, digit_labels)
            
            #print(loss_left, loss_right)
            
            loss = criterion(output.flatten(), labels.float().flatten())
            
            aux_loss = ALPHA*loss + BETA*loss_left + GAMMA*loss_right

            loss.require_grad = True
            loss.backward()

            optimizer.step()

            if ind_batch % 100 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, ind_batch, len(dataloader), loss
                    )
                )
                
            if save_model_flag and (epoch-1 % SAVE_MODEL_EVERY_X_EPOCH == 0):
                save_model(
                    model=model, epoch=epoch, loss=loss.item(), save_dir=checkpoints_dir
                )
                print(f"model saved to {str(checkpoints_dir)}")
               
            
def train_single(model,dataloader,epochs, learning_rate, criterion, compare= True, save_model_flag=False):
    
    model.train() #set the model on training mode 
    
    print("Training with {0} epochs, a learning rate of {1} and {2} as loss function".format(epochs,learning_rate, criterion))
    
    #put on the gpu if available 
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("CUDA available")
    else:
        print("NO CUDA")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []
    
    for epoch in range(1, epochs+1):
        sum_loss_epoch = 0
        
        for ind_batch, sample_batched in enumerate(dataloader):
    
            images = sample_batched["images"]
            
            #choose the output
            if(compare):
                labels = sample_batched["bool_labels"]
            else:
                labels = sample_batched["digit_labels"]
           
            if cuda:
                images = images.to(device="cuda")
                labels = labels.to(device="cuda")

            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels.float().view(-1,1))

            loss.require_grad = True
            loss.backward()

            optimizer.step()

            if ind_batch % 100 == 0:
                print("[Epoch {}, Batch {}/{}]:  [Loss: {:0.2f}]".format(epoch, ind_batch, len(dataloader), loss))
                
            if save_model_flag and (epoch-1 % SAVE_MODEL_EVERY_X_EPOCH == 0):
                save_model(
                    model=model, epoch=epoch, loss=loss.item(), save_dir=checkpoints_dir
                )
                print(f"model saved to {str(checkpoints_dir)}")
            
            sum_loss_epoch = sum_loss_epoch + loss.item()
        
        loss_epoch = sum_loss_epoch / float(len(dataloader))
        print("At epoch {0} the total loss is {1}".format(epoch, loss_epoch) )
        training_losses.append(loss_epoch)
        
    return training_losses

def predict_test(model,dataloader,criterion, compare = True):
    cuda = torch.cuda.is_available()
    
    model.eval() # set the model on evaluation mode
    sum_loss = 0
    total = 0
    correct = 0
    for ind_batch, sample_batched in enumerate(dataloader):
      
        images = sample_batched["images"]
            
        #choose the output
        if(compare):
            labels = sample_batched["bool_labels"]
        else:
            labels = sample_batched["digit_labels"]
           
        if cuda:
            images = images.to(device="cuda")
            labels = labels.to(device="cuda")
                
        output = model(images)
        
        labels= labels.float().view(-1,1)
        
        sum_loss += criterion(output, labels)

        total += labels.size(0)  
        correct += (output.round() == labels).sum()  
    
    test_loss = sum_loss / float(len(dataloader))
    accuracy = 100 * correct / float(total)
        
    return test_loss, accuracy