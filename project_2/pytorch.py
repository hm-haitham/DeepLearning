import torch.nn as nn
import torch.optim as opt
import time
import torch
import math


def train_model_with_pytorch(model, X_train, y_train, X_test, y_test, epochs, mini_batch_size , 
                lr = 0.01, momentum = 0.9):
    
    total_loss = []
    total_acc = []
    
    test_pred = [] 
    test_loss = []
    test_acc = []
    
    criterion = nn.MSELoss()
    optimizer = opt.SGD(model.parameters(), lr, momentum=momentum)
    
    #one hot encoding 
    encoded_y_train = one_hot_encoding(y_train)
    
    number_batches =  X_train.size(0) // mini_batch_size
    
    for epoch in range(epochs):
        
        loss_epoch = 0
        acc_epoch = 0
        pred = []
        
        for index in range(0, X_train.size(0), mini_batch_size):
            optimizer.zero_grad()
            X_train_batched = X_train[index:(index+mini_batch_size)]
            
            y_train_batched = y_train[index:(index+mini_batch_size)]
            y_train_encoded_batched = encoded_y_train[index:(index+mini_batch_size)]
            
            #forward pass
            output = model(X_train_batched)
            loss = criterion(output, y_train_encoded_batched)
            loss_epoch += loss.item() 
            
            #batch_acc = ((output.round() == y_train_encoded_batched_).sum(1) == 2).sum(0)
            batch_acc =  (output.max(1)[1].float() == y_train_batched).sum()
            acc_epoch += batch_acc
            
            #model.zero_grad()
            #backward pass and update parameters gradient
            loss.backward()  
            #update parameters
            optimizer.step()
        
        total_acc.append(acc_epoch)
        total_loss.append(loss_epoch / number_batches)
        #compute the predictions on the test set for this epoch model
        pred_epoch, test_loss_epoch, test_acc_epoch = predict(model, X_test, y_test,criterion)
        if (epoch %10 ==0):
            print(epoch,'. Obtain on the test set an average loss of {0} and an accuracy of {1}'.format(test_loss_epoch,test_acc_epoch))
    
        test_pred.append(pred_epoch)
        test_loss.append(test_loss_epoch)
        test_acc.append(test_acc_epoch)

    return model, total_loss, total_acc, test_pred, test_loss, test_acc 


def predict_with_torch(model, X_test, y_test, criterion, mini_batch_size = 1):
    
    number_batches =  X_train.size(0) // mini_batch_size
    #one hot encoding 
    encoded_y_train = one_hot_encoding(y_train)
    
    pred = []
    total_loss = 0
    total_acc = 0
    
    for index in range(0, X_test.size(0), mini_batch_size):
        X_test_batched = X_test[index:(index+mini_batch_size)]
            
        y_test_batched = y_test[index:(index+mini_batch_size)]
        y_test_encoded_batched = encoded_y_test[index:(index+mini_batch_size)]
        
        #forward pass
        output = model(X_test_batched)
        
        loss = criterion(output, y_test_encoded_batched)
        total_loss += loss.item() 
        
        batch_acc =  (output.max(1)[1].float() == y_test_batched).sum()
        total_acc += batch_acc.item()
        pred.append(output)
        
    total_loss = total_loss / number_batches
    

    return pred, total_loss,total_acc


torch.set_grad_enabled(True)
torch.manual_seed(0)
np.random.seed(0)

rounds = 10

input_units = 2
hidden_units = 25
output_units = 2

epochs = 50
mini_batch_size = 5

X_train, y_train = build_data(1000)  #(1000,2)
X_test, y_test = build_data(1000)  #(1000,2) 

print('Start training with parameters : {0} rounds, {1} epochs and {2} batch size'.format(rounds,epochs,mini_batch_size))

result_rounds = []  #training_losses, training_acc, test_losses, test_acc

time1 = time.perf_counter()
for i in range(rounds):     
        
    print("Training round {0} : ".format(i))
    model_torch = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_units),
            nn.Sigmoid()
    )
    
    #array of shape (rounds,epochs)
    model_trained, train_loss, train_acc, test_pred, test_loss, test_acc = train_model_with_pytorch(model_torch, X_train, y_train, X_test, y_test, epochs, mini_batch_size , lr = 0.01, momentum = 0.9)
    
    
    result_rounds.append([train_loss, train_acc, test_loss, test_acc])

time2 = time.perf_counter()  
print('Time it took to train {0} rounds : {1} '.format(rounds, (time2 - time1) ) )

results = np.array(result_rounds).mean(axis=0) 
train_loss_epochs = results[0]
test_loss_epochs = results[2]
    
train_acc_epochs = results[1]
test_acc_epochs = results[3]

print('After training on {0} rounds we get on average :'.format(rounds))

for i in [0,9,19,29,39,49]:
    print('On epoch {0} : We get on the train set an average loss of {1} and an accuracy of {2}'.format(i+1,train_loss_epochs[i],train_acc_epochs[i]))
    print('On epoch {0} : We get on the test set an average loss of {1} and an accuracy of {2}'.format(i+1,test_loss_epochs[i],test_acc_epochs[i]))
        