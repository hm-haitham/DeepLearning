from loss.LossMSE import LossMSE
from loss.LossBCE import LossBCE
from utils import one_hot_encoding

def predict(model, X_test, y_test, mini_batch_size = 1, loss_name = 'MSE'):
    
    number_batches =  X_test.size(0) // mini_batch_size

    #one hot encoding 
    encoded_y_test = one_hot_encoding(y_test)
    
    pred = []
    total_loss = 0
    total_acc = 0
    
    if(loss_name == 'BCE'):
        criterion = LossBCE(model)
    else: 
        criterion = LossMSE(model)
        
    for index in range(0, X_test.size(0), mini_batch_size):
        X_test_batched = X_test[index:(index+mini_batch_size)]
            
        y_test_batched = y_test[index:(index+mini_batch_size)]
        y_test_encoded_batched = encoded_y_test[index:(index+mini_batch_size)]
        
        #forward pass
        output = model.forward(X_test_batched)
        
        #compute test loss
        loss = criterion.forward(output, y_test_encoded_batched)
        total_loss += loss.item() 
        
        #compute test accuracy
        batch_acc =  (output.max(1)[1].float() == y_test_batched).sum()
        total_acc += batch_acc.item()
        
        #predictions
        pred.append(output)
        
    total_loss = total_loss / number_batches
    
    return pred, total_loss, total_acc