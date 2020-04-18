from torch.utils.data import Dataset


class PairDataset(Dataset):

    def __init__(self, data, bool_labels, digit_labels = None):
        self.images = data
        
        self.bool_labels = bool_labels
        
        if digit_labels is not None:
            self.digit_labels = digit_labels

    def __len__(self):
        # override the class method. return the length of data
        return len(self.bool_labels)

    def __getitem__(self, idx):
        # override the class method. return the item at the index(idx)
        if self.digit_labels is not None:
            sample = {"images" : self.images[idx],
                      "bool_labels" : self.bool_labels[idx],
                      "digit_labels" : self.digit_labels[idx]}
        else:
            sample = {"images" : self.images[idx],
                      "bool_labels" : self.bool_labels[idx]}
            
        return sample
    
class SingleDataset(Dataset):

    def __init__(self, data, digit_labels):
        self.images = data
        
        self.digit_labels = digit_labels

    def __len__(self):
        # override the class method. return the length of data
        return len(self.digit_labels)

    def __getitem__(self, idx):
        # override the class method. return the item at the index(idx)
        sample = {"images" : self.images[idx],
                  "digit_labels" : self.digit_labels[idx]}
            
        return sample