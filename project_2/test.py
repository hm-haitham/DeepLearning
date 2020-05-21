from linear import Linear
from sequential import Sequential

from activation_function.tanh import Tanh
from activation_function.sigmoid import Sigmoid
from activation_function.relu import ReLU
from activation_function.leakyReLu import LeakyReLU

from optimizer.sgd import SGD
from optimizer.momentum_sgd import MomentumSGDOptimizer
from optimizer.rmsprop import RMSPropOptimizer
from optimizer.adam import AdamOptimizer

from loss.LossMSE import LossMSE
from loss.LossBCE import LossBCE

from utils import build_data, one_hot_encoding
from predict import predict
import numpy as np 
import torch, time

def train_model(model, X_train, y_train, X_test, y_test, epochs, mini_batch_size , 
                lr = 0.01, momentum = 0.9, gamma = 0.9, epsilon = 1e-8, beta1 = 0.9,
                beta2 = 0.999, loss_name = 'MSE', opt = 'ADAM'):
    
    total_loss = []
    total_acc = []
    
    test_pred = [] 
    test_loss = []
    test_acc = []
    
    if(loss_name == 'BCE'):
        criterion = LossBCE(model)
    else: 
        criterion = LossMSE(model)
    
    if(opt == 'SGD'):
        optimizer = SGD(model, lr)
    elif(opt == 'MOMENTUM'):
        optimizer = MomentumSGDOptimizer(model, lr, momentum) 
    elif(opt == 'RMS'):
        optimizer = RMSPropOptimizer(model, lr, gamma, epsilon)
    else: 
        optimizer = AdamOptimizer(model, lr, beta1, beta2, epsilon)
    
    #one hot encoding 
    encoded_y_train = one_hot_encoding(y_train)
    
    number_batches =  X_train.size(0) // mini_batch_size
    
    for epoch in range(epochs):
        
        loss_epoch = 0
        acc_epoch = 0
        pred = []
        
        for index in range(0, X_train.size(0), mini_batch_size):
            X_train_batched = X_train[index:(index+mini_batch_size)]
            
            y_train_batched = y_train[index:(index+mini_batch_size)]
            y_train_encoded_batched = encoded_y_train[index:(index+mini_batch_size)]
            
            #forward pass
            output = model.forward(X_train_batched)
            loss = criterion.forward(output, y_train_encoded_batched)
            loss_epoch += loss.item() 
            
            batch_acc =  (output.max(1)[1].float() == y_train_batched).sum()
            acc_epoch += batch_acc
            
            #backward pass and update parameters gradient
            criterion.backward()  
            #update parameters
            optimizer.step()
        
        total_acc.append(acc_epoch)
        total_loss.append(loss_epoch / number_batches)
        
        #compute the predictions on the test set for this epoch model
        pred_epoch, test_loss_epoch, test_acc_epoch = predict(model, X_test, y_test, loss_name = loss_name)
        
        test_pred.append(pred_epoch)
        test_loss.append(test_loss_epoch)
        test_acc.append(test_acc_epoch)
        
        if(( (epoch+ 1) %10 == 0 ) or (epoch == 0) ) :
            print('On epoch {0} : We get on the train set an average loss of {1} and an accuracy of {2}'.format(epoch+1,loss_epoch / number_batches,acc_epoch))
            print('On epoch {0} : We get on the test set an average loss of {1} and an accuracy of {2}'.format(epoch+1,test_loss_epoch,test_acc_epoch))
        
    return model, total_loss, total_acc, test_pred, test_loss, test_acc 

###############################################################################################################################

#For reproducibility
torch.set_grad_enabled(False)
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
        
    print("Training round {0} : ".format(i+1))
    model = Sequential(
            Linear(input_units, hidden_units),
            ReLU(),
            Linear(hidden_units, hidden_units),
            ReLU(),
            Linear(hidden_units, hidden_units),
            ReLU(),
            Linear(hidden_units, output_units),
            Sigmoid()
            )
    #array of shape (rounds,epochs)
    model_trained, train_loss, train_acc, test_pred, test_loss, test_acc = train_model(model, X_train, y_train, X_test, y_test,
                                                                           epochs, mini_batch_size , lr = 0.01, opt = 'SGD',loss_name = 'MSE')
    
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
        