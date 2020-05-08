import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from predict import predict_siamese, predict_basic

from config import EPOCHS
from config import FINAL_CRITERION
from config import LEARNING_RATE
from config import SUB_CRITERION
from config import WEIGHTS_LOSS

def train_siamese(model, 
                  dataloader, 
                  test_dataloader,
                  epochs = EPOCHS,
                  final_criterion = FINAL_CRITERION, 
                  learning_rate = LEARNING_RATE,
                  aux_loss = False,
                  sub_criterion = SUB_CRITERION, 
                  weights_loss = WEIGHTS_LOSS):
    
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("CUDA available")
    else:
        print("NO CUDA")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    training_acc = []
    
    training_losses_l = []
    
    training_losses_r = []
    
    test_losses = []
    test_acc = []
    
    test_losses_l = []
    
    test_losses_r = []

    for epoch in range(1, epochs+1):  
        model.train()
        
        sum_loss_epoch = 0
        total = 0
        correct = 0
        accuracy_epoch = 0
        
        sum_loss_epoch_l = 0
        
        sum_loss_epoch_r = 0
        
        for ind_batch, sample_batched in enumerate(dataloader):
            
            images = sample_batched["images"]
            labels = sample_batched["bool_labels"]
            digit_labels = sample_batched["digit_labels"]
            
            labels = labels.unsqueeze(1)
            
            if cuda:
                images = images.to(device="cuda")
                labels = labels.to(device="cuda")

            optimizer.zero_grad()
                       
            output, lefted, righted = model(images)
            
            loss = final_criterion(output.flatten(), labels.float().flatten())
            loss_left = sub_criterion(lefted, digit_labels[:,0])
            loss_right = sub_criterion(righted, digit_labels[:,1])
            
            if aux_loss:
                alpha, beta, gamma = weights_loss
                loss = alpha * loss + beta * loss_left + gamma * loss_right

            loss.require_grad = True
            loss.backward()

            optimizer.step()
            
            #update the accuracy 
            total += images.size(0)  
            correct += (output.round() == labels).sum() 
            
            if ind_batch % 250 == 0:
                print("[Epoch {}, Batch {}/{}]:  [Loss: {:.2f}]".format(epoch, ind_batch, len(dataloader), loss) )
                
            #add the loss for this batch to the total loss of the epoch
            sum_loss_epoch = sum_loss_epoch + loss.item()
            sum_loss_epoch_l = sum_loss_epoch_l + loss_left.item()
            sum_loss_epoch_r = sum_loss_epoch_r + loss_right.item()
            
            
        #compute the mean to obtain the loss for this epoch 
        mean_loss = sum_loss_epoch / float(len(dataloader))
        mean_loss_l = sum_loss_epoch_l / float(len(dataloader))
        mean_loss_r = sum_loss_epoch_r / float(len(dataloader))
        
        print("At epoch {0} the training loss is {1}".format(epoch, mean_loss) )
        training_losses.append(mean_loss)
        
        accuracy_epoch = float(correct) / float(total)
        print("At epoch {0} the training accuracy is {1}".format(epoch, accuracy_epoch) )
        training_acc.append(accuracy_epoch)
        
        training_losses_l.append(mean_loss_l)
        training_losses_r.append(mean_loss_r)
        
        print("At epoch {0} :".format(epoch))
        
        test_loss, test_accuracy, test_loss_l, test_loss_r = predict_siamese(model,
                                                                     test_dataloader,
                                                                     final_criterion,
                                                                     aux_loss,
                                                                     sub_criterion,
                                                                     weights_loss)
        
        test_losses.append(test_loss)
        test_acc.append(test_accuracy)
        test_losses_l.append(test_loss_l)
        test_losses_r.append(test_loss_r)
        
    return training_losses, training_acc, training_losses_l, training_losses_r, test_losses, test_acc, test_losses_l, test_losses_r

def train_basic(model, 
                dataloader, 
                test_dataloader,
                epochs = EPOCHS, 
                final_criterion = FINAL_CRITERION, 
                learning_rate=LEARNING_RATE):
    
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("CUDA available")
    else:
        print("NO CUDA")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []
    training_acc = []
    
    test_losses = []
    test_acc = []

    for epoch in range(1, epochs+1):  
        sum_loss_epoch = 0
        total = 0
        correct = 0
        accuracy_epoch = 0
        
        for ind_batch, sample_batched in enumerate(dataloader):
            
            images = sample_batched["images"]
            labels = sample_batched["bool_labels"]
            
            if cuda:
                images = images.to(device="cuda")
                labels = labels.to(device="cuda")

            optimizer.zero_grad()

            output = model(images)
            
            loss = final_criterion(output.flatten(), labels.float().flatten())

            loss.require_grad = True
            loss.backward()

            optimizer.step()

            #update the accuracy 
            total += images.size(0)  
            correct += (output.round() == labels).sum() 
            
            if ind_batch % 250 == 0:
                print("[Epoch {}, Batch {}/{}]:  [Loss: {:.2f}]".format(epoch, ind_batch, len(dataloader), loss) )
                
            #add the loss for this batch to the total loss of the epoch
            sum_loss_epoch = sum_loss_epoch + loss.item()
            
        #compute the mean to obtain the loss for this epoch 
        mean_loss = sum_loss_epoch / float(len(dataloader))
        
        print("At epoch {0} the loss is {1}".format(epoch, mean_loss) )
        training_losses.append(mean_loss)
        
        accuracy_epoch = float(correct) / float(total)
        print("At epoch {0} the accuracy is {1}".format(epoch, accuracy_epoch) )
        training_acc.append(accuracy_epoch)
        
        test_loss, test_accuracy = predict_basic(model,
                                                 test_dataloader,
                                                 final_criterion)

        test_losses.append(test_loss)
        test_acc.append(test_accuracy)
        
    return training_losses, training_acc, test_losses, test_acc