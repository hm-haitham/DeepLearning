import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from config import TRAIN_CHECKPOINTS_DIR
from config import LEARNING_RATE

def save_model(model, epoch=None, loss=None, save_dir=None, specific_name=None):
    """Saves a checkpoint of the model state

        Args:
            model (nn.Module) : Model to be saved

            epoch : The epoch of this checkpoint
            
            loss : The loss of the last batch of this checkpoint
            
            save_dir : path for where to save the checkpoint
            
            specifi_name : a specific name to give to the checkpoint

        """
    if epoch and loss and save_dir and specific_name:
        model_name = model.model_name
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"{timestr}_{model_name}_epoch_{epoch}_loss_{loss:03.3f}.pt"
        Path(save_dir).mkdir(exist_ok=True)
        file_path = Path(save_dir) / file_name
        torch.save(model.state_dict(), str(file_path))
    elif save_dir and specific_name:
        file_path = Path(save_dir) / specific_name
        torch.save(model.state_dict(), str(file_path))

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
            labels = sample_batched["bool_labels"]
            
            if cuda:
                images = images.to(device="cuda")
                labels = labels.to(device="cuda")

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output.flatten(), labels.float().flatten())

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