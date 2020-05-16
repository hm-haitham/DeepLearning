import torch
import torch.nn as nn
import torch.utils.data as data
import config

def predict_siamese(model, dataloader, aux_loss = False, alpha = config.ALPHA):
    
    model.eval()
    
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        
    sum_loss = 0
    total = 0
    correct = 0
    accuracy = 0

    sum_loss_l = 0
    sum_loss_r = 0

    final_criterion = nn.BCELoss()
    sub_criterion = nn.CrossEntropyLoss()
    
    for ind_batch, sample_batched in enumerate(dataloader):

        images = sample_batched["images"]
        labels = sample_batched["bool_labels"].float().view(-1,1)
        digit_labels = sample_batched["digit_labels"]
        
        if cuda:
            images = images.to(device="cuda")
            labels = labels.to(device="cuda")
            digit_labels = digit_labels.to(device="cuda")

        output, lefted, righted = model(images)
        
        loss = final_criterion(output, labels)
        loss_left = sub_criterion(lefted, digit_labels[:,0])
        loss_right = sub_criterion(righted, digit_labels[:,1])

        if aux_loss:
            loss = alpha * loss + ((1-alpha)/2) * loss_left + ((1-alpha)/2) * loss_right

        #update the accuracy 
        total += images.size(0)  
        correct += (output.round() == labels).sum() 

        #add the loss for this batch to the total loss of the epoch
        sum_loss = sum_loss + loss.item()
        sum_loss_l = sum_loss_l + loss_left.item()
        sum_loss_r = sum_loss_r + loss_right.item()

    #compute the mean to obtain the loss for this epoch 
    mean_loss = sum_loss / float(len(dataloader))
    mean_loss_l = sum_loss_l / float(len(dataloader))
    mean_loss_r = sum_loss_r / float(len(dataloader))
    
    print("The test loss is {0}".format(mean_loss) )

    accuracy = float(correct) / float(total)
    print("The test accuracy is {0}".format(accuracy) )
        
    return mean_loss, accuracy, mean_loss_l, mean_loss_r

def predict_basic(model, dataloader):
    
    model.eval()
    
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        
    sum_loss = 0
    total = 0
    correct = 0
    accuracy = 0
    
    final_criterion = nn.BCELoss()
    
    for ind_batch, sample_batched in enumerate(dataloader):

        images = sample_batched["images"]
        labels = sample_batched["bool_labels"].float().view(-1,1)
        
        if cuda:
            images = images.to(device="cuda")
            labels = labels.to(device="cuda")

        output = model(images)
        
        loss = final_criterion(output, labels)

        #update the accuracy 
        total += images.size(0)  
        correct += (output.round() == labels).sum() 

        #add the loss for this batch to the total loss of the epoch
        sum_loss = sum_loss + loss.item()

    #compute the mean to obtain the loss for this epoch 
    mean_loss = sum_loss / float(len(dataloader))
    
    print("The test loss is {0}".format(mean_loss) )

    accuracy = float(correct) / float(total)
    print("The test accuracy is {0}".format(accuracy) )
        
    return mean_loss, accuracy