import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from helpers import save_model

from config import TRAIN_CHECKPOINTS_DIR
from config import LEARNING_RATE

def train_double(
    model,
    dataloader,
    epochs,
    criterion,
    save_model_flag=False,
):
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
               
            
def train_single(
    model,
    dataloader,
    epochs,
    criterion,
    save_model_flag=False,
):
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
                
            if save_model_flag and (epoch-1 % SAVE_MODEL_EVERY_X_EPOCH == 0):
                save_model(
                    model=model, epoch=epoch, loss=loss.item(), save_dir=checkpoints_dir
                )
                print(f"model saved to {str(checkpoints_dir)}")