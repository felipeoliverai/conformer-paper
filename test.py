from requests import head
import torch 
from torch import nn


# check out GPU 
device = torch.cuda.is_available()
print(device)



#x = torch.tensor([[34, 56, 678, 10], [10, 46, 79, 10], [79, 98, 34, 5]])
#print(x)


torch.manual_seed(42)
encoder_dim = 256
inputs = torch.rand(256)

# transpose matrix 

matrix = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print("------- Matrix --------\n")
print(matrix)

transpose_matrix = torch.t(matrix)
print("\n\n -------- Transpose Matrix --------- \n")
print(transpose_matrix)



#q_value = nn.Linear(encoder_dim, encoder_dim, bias=False)
#k_value = nn.Linear(encoder_dim, encoder_dim, bias=False)
#v_value = nn.Linear(encoder_dim, encoder_dim, bias=False)

#n_heads = 5

#att_q  = []
#att_k = []
#att_v = []

#for _ in range(0, n_heads):
    
#    q_tensor = q_value.forward(inputs)
#    k_tensor = k_value.forward(inputs)
#    v_tensor = v_value.forward(inputs)

    #att_list.append([q_tensor,  k_tensor,  v_tensor])
#    att_q.append(q_tensor)
#    att_k.append(k_tensor)
#    att_v.append(q_tensor)


#q_convert = torch.stack(att_q)
#k_convert = torch.stack(att_k)
#v_convert = torch.stack(att_v)


#concat = torch.cat((q_convert, k_convert, v_convert))
#print(concat)