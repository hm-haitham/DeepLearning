import torch.optim as optim

def train_ann(model,dataloader,epochs, learning_rate, criterion, compare= True, save_model_flag=False):
    
    model.train() #set the model on training mode 
    
    print("Training with {0} epochs, learning rate of {1} and {2} as loss function".format(epochs,learning_rate, criterion))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []
    training_acc = []
    
    for epoch in range(1, epochs+1):
        
        total = 0
        correct = 0
        sum_loss_epoch = 0
        
        for ind_batch, sample_batched in enumerate(dataloader):
            
            images = sample_batched["images"]
            
            #choose the output
            if(compare):
                labels = sample_batched["bool_labels"]
            else:
                labels = sample_batched["digit_labels"]
            
            labels = labels.float().view(-1,1)
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.require_grad = True
            loss.backward()

            optimizer.step()
            
            #update the accuracy 
            total += images.size(0) 
            correct += (output.round() == labels).sum() 
            
            if ind_batch % 250 == 0:
                print("[Epoch {}, Batch {}/{}]:  [Loss: {:0.2f}]".format(epoch, ind_batch, len(dataloader), loss))
        
            sum_loss_epoch = sum_loss_epoch + loss.item()
        
        loss_epoch = sum_loss_epoch / float(len(dataloader))
        print("At epoch {0} the total loss is {1}".format(epoch, loss_epoch) )
        training_losses.append(loss_epoch)
        
        accuracy_epoch = float(correct) / float(total)
        print("At epoch {0} the accuracy is {1}".format(epoch, accuracy_epoch) )
        training_acc.append(accuracy_epoch)
        
    return training_losses, training_acc

def test_ann(model,dataloader, criterion, compare = True):

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
     
        output = model(images)
        
        labels = labels.float().view(-1,1)
        
        sum_loss += criterion(output, labels)

        total += labels.size(0)  
        correct += (output.round() == labels).sum()  
    
    test_loss = sum_loss.item() / float(len(dataloader))
    accuracy =  float(correct) / float(total)
        
    return test_loss, accuracy