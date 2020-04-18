import torch

from config import TEST_BATCH_SIZE
from config import NB_SAMPLES
from helpers import compute_accuracy

def predict(model, dataloader, model_weights=None):
    """Make predictions based on the given model, and saves images

        Args:
            model (nn.Module) : Model to predict with
            
            dataloader : The dataloader of the test set

            model_weights : a dict containing the state of the weights with which to initialize our model

        """

    if model_weights is not None:
        model.load_state_dict(torch.load(str(model_weights)))

    model.eval()

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("CUDA is available")
    else:
        print("CUDA is NOT available")

    result = torch.zeros(NB_SAMPLES)
    true_labels = torch.zeros(NB_SAMPLES)
    
    for ind_batch, sample_batched in enumerate(dataloader):
        batch_images = sample_batched["images"]
        
        current_batch_size = batch_images.size(0)
        
        if cuda:
            batch_images = batch_images.to(device="cuda")
        with torch.no_grad():
            output = model(batch_images)
        
        final = output.clone().detach().cpu()
        final[final > 0.5] = 1
        final[final <= 0.5] = 0
        
        result[ind_batch*TEST_BATCH_SIZE : ind_batch*TEST_BATCH_SIZE + current_batch_size] = final.flatten()
        true_labels[ind_batch*TEST_BATCH_SIZE : ind_batch*TEST_BATCH_SIZE + current_batch_size] = sample_batched["bool_labels"].float().flatten()
        
        if ind_batch % 100 == 0:
            print("[Batch {}/{}]".format(ind_batch, len(dataloader)))
            
    accuracy = compute_accuracy(result, true_labels)
            
    return result, accuracy
