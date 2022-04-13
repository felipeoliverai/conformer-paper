import torch 
import torch.nn as nn


print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class Swish(nn.Module):

    """
       Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks
       applied to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x:torch.Tensor):
        return  x * torch.sigmoid(x)




class FeedForwardModule(nn.Module):

    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.
    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module. 
        
    """


    def __init__(self, encoder_dim = 512, expansion_factor = 4, dropout_p = 0.1):
        super(FeedForwardModule, self).__init__()

        self.sequential = nn.Sequential(
                          nn.LayerNorm(encoder_dim),
                          nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
                          Swish(), 
                          nn.Dropout(p=dropout_p), 
                          nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
                          nn.Dropout(p=dropout_p)

        )

    
    def forward(self, x): 
        return self.sequential(x)


# Testing FeedForward

torch.manual_seed(42)
inputs = torch.rand(512, device=device)
#print(inputs.device)

print("\n\n")
model = FeedForwardModule(encoder_dim=512, expansion_factor=4, dropout_p=0.1).to(device)
#print(model)
print(model.forward(inputs)) 
