import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from config import TRAIN_CHECKPOINTS_DIR
from config import LEARNING_RATE

def train(
    model,
    dataloader,
    epochs,
    criterion,
    save_model_flag=False,
    model_weights=None,
    checkpoints_dir=TRAIN_CHECKPOINTS_DIR,
    last_checkpoint=None,
):
    """Trains the model

        Args:
            model (nn.Module) : Model to be trained
            
            dataloader : The dataloader of the training set

            epochs (int) : Number of epochs to train the model
            
            criterion : Loss function for the training
            
            checkpoints_dir : path for where to save the checkpoints
            
            model_weights : a dict containing the state of the weights with which to initialize our model

        """

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
            labels = sample_batched["digit_labels"]
           
            if cuda:
                images = images.to(device="cuda")
                labels = labels.to(device="cuda")

            optimizer.zero_grad()
            
            output = model(images)

            loss = criterion(output, labels.float())

            loss.require_grad = True
            loss.backward()

            optimizer.step()

            if ind_batch % 100 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, ind_batch, len(dataloader), loss
                    )
                )