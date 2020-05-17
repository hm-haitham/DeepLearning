import config

from helpers import generate_pair_sets
from datasets import PairDataset

from models.FCN import FCN
from models.CNN import CNN
from models.SiameseNet import SiameseNet

from train import train_siamese
import torch.utils.data as data
import torch 
import numpy as np 

def compute_results(subnet_type, weight_sharing, aux_loss, rounds = 1 ):
    
    print("=" * 100)
    print('\nWeight sharing:', weight_sharing, '   Aux loss:', aux_loss,'       Subnet:', subnet_type)
    
    if subnet_type == 'FCN' :
        alpha = config.BEST_ALPHA_FCN
        nb_hidden_layers = config.SIAMESE_NET_BEST_NB_FCN
        hidden_layer = config.SIAMESE_NET_BEST_HIDDEN_FCN
        
        print('\nSubnet parameters:', '   HL: ',config.FCN_BEST_HIDDEN, '  KL: ', config.FCN_BEST_NB)
        print('\nSiamese parameters:', '   HL: ',hidden_layer, '  KL: ', nb_hidden_layers,
              '  alpha: ',alpha)
        
        subnet1 = FCN(nb_hidden_layers = config.FCN_BEST_NB, hidden_layer = config.FCN_BEST_HIDDEN)
        subnet2 = FCN(nb_hidden_layers = config.FCN_BEST_NB, hidden_layer = config.FCN_BEST_HIDDEN)
    else :
        alpha = config.BEST_ALPHA_CNN
        nb_hidden_layers = config.SIAMESE_NET_BEST_NB_CNN
        hidden_layer = config.SIAMESE_NET_BEST_HIDDEN_CNN
        
        print('\nSubnet parameters:','   BC: ', config.CNN_BEST_CHANNEL,'   KC: ',config.CNN_BEST_KERNEL_SIZE,
              '   HL: ',config.CNN_BEST_HIDDEN, '  KL: ', config.CNN_BEST_NB)
        print('\nSiamese parameters:', '   HL: ',hidden_layer, '  KL: ', nb_hidden_layers,
              '  alpha: ' ,alpha)
        
        subnet1 = CNN(nb_hidden_layers = config.CNN_BEST_NB, hidden_layer = config.CNN_BEST_HIDDEN,
                  base_channel_size = config.CNN_BEST_CHANNEL, kernel_size = config.CNN_BEST_KERNEL_SIZE)
        subnet2 = CNN(nb_hidden_layers = config.CNN_BEST_NB, hidden_layer = config.CNN_BEST_HIDDEN,
                  base_channel_size = config.CNN_BEST_CHANNEL, kernel_size = config.CNN_BEST_KERNEL_SIZE)

    if weight_sharing:
        model = SiameseNet(subnet1, nb_hidden_layers = nb_hidden_layers, hidden_layer = hidden_layer)
    else:
        model = SiameseNet(subnet1, subnet2, nb_hidden_layers = nb_hidden_layers, hidden_layer = hidden_layer)
    
    results_test = []
    results_train = []
    
    for i in range(rounds):
        print('\nTrain beginning...')
        training_losses, training_acc, _, _, test_losses, test_acc, _, _ = train_siamese(
                    model = model, dataloader = train_dataloader, test_dataloader = test_dataloader, aux_loss = aux_loss,
                    alpha = alpha)

        print('\nTrain complete !')

        final_train_loss, final_train_loss_acc = training_losses[-1], training_acc[-1]
        results_train.append([final_train_loss, final_train_loss_acc])
        
        final_test_loss, final_test_loss_acc = test_losses[-1], test_acc[-1]
        results_test.append([final_test_loss, final_test_loss_acc])
        
    mean_train_results = np.array(results_train).mean(axis=0)
    print("In epoch {0}, on the train set we obtain a loss of {1} and an accuracy of {2}".format(config.EPOCHS, 
                                                                                                    round(mean_train_results[0], 3),
                                                                                                    round(mean_train_results[1], 3)))

    mean_test_results = np.array(results_test).mean(axis=0)
    print("In epoch {0}, on the test set we obtain a loss of {1} and an accuracy of {2}".format(config.EPOCHS, 
                                                                                                    round(mean_test_results[0], 3),
                                                                                                    round(mean_test_results[1], 3)))
    
###############################################################################################################################

#For reproducibility
torch.manual_seed(1)
np.random.seed(1)

print("Training 8 different architectures with optimal parameters :")
print("By default 1 round can be modified")

pairs = generate_pair_sets(config.NB_SAMPLES)

train_dataset = PairDataset(pairs[0], pairs[1], pairs[2])
train_dataloader = data.DataLoader(dataset=train_dataset, batch_size= config.TRAIN_BATCH_SIZE, shuffle=True)

test_dataset = PairDataset(pairs[3], pairs[4], pairs[5])
test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=True)

configuration = [
    (weight_sharing, aux_loss)
    
    for weight_sharing in [False,True]
    for aux_loss in [False,True]
]
configuration
    
for weight_sharing, aux_loss in configuration :
    compute_results('FCN', weight_sharing, aux_loss)

for weight_sharing, aux_loss in configuration :
        compute_results('CNN', weight_sharing, aux_loss)
    
print("=" * 100)
        
        