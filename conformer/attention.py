import torch 
import torch.nn as nn 



class PositionalEncoding(nn.Module):

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    positional = 1 # testing
    tensor = torch.tensor([45, 67, 90])