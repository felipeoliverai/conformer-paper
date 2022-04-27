from requests import head
import torch 
from torch import nn


#x = torch.tensor([[34, 56, 678, 10], [10, 46, 79, 10], [79, 98, 34, 5]])
#print(x)


torch.manual_seed(42)
encoder_dim = 256
inputs = torch.rand(256)


q_value = nn.Linear(encoder_dim, encoder_dim, bias=False)
k_value = nn.Linear(encoder_dim, encoder_dim, bias=False)
v_value = nn.Linear(encoder_dim, encoder_dim, bias=False)

n_heads = 5
heads = torch.empty((256))

for _ in range(1, n_heads):
    
    q_tensor = q_value.forward(inputs)
    k_tensor = k_value.forward(inputs)
    v_tensor = v_value.forward(inputs)
    concat = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

    #heads.append(concat)


print(concat.size())
#print(heads)
#heads_tensor = torch.Tensor(heads)
#print(type(heads_tensor))




#print(x, "\n\n")
#print(x.size())

#print(type(x))
#print("\n\n Transpose: ", torch.t(x))