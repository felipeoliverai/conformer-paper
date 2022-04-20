from turtle import forward
import torch 
import torch.nn as nn 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ScaledDotProductAttention(nn.Module):

    """ 
        scaled dot-product attention: the output is a weighted sum of the values,
        where the weight assigned to each value is determined by the dot-product of the query with all the keys.

    """

    def __init__(self, q_value, k_value, v_value):
        super(ScaledDotProductAttention, self).__init__()

        # Scaled Dot-Product Attention (Self-attention)
        self.q_value = q_value
        self.k_value = k_value
        self.v_value = v_value

    def forward(self, q_value, k_value, v_value, scale=None):
        
        # matmul (q * kT) "t represents a tranpose matrix"
        q_k_matmul = torch.matmul(q_value, k_value) * v_value # to fix 

        # softmax ()



#class PositionalEmbedding(nn.Module)



class MHSA(nn.Module):

    def __init__(self, encoder_dim = 256):
        super(MHSA, self).__init__()

        # Q,K,V vectors (linear transformations)
        self.fc_q = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.fc_k = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.fc_value = nn.Linear(encoder_dim, encoder_dim, bias=False)


    def forward(self, query, key, value, mask=None):

        # linear Q,K,V
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Scaled Dot Product layer 
        attention = ()
        




torch.manual_seed(42)
inputs = torch.rand(256, device=device)
#model = MHSA(encoder_dim=256).to(device)
#print(model)
encoder_dim = 256

q_value = nn.Linear(encoder_dim, encoder_dim, bias=False).to(device)
k_value = nn.Linear(encoder_dim, encoder_dim, bias=False).to(device)
v_value = nn.Linear(encoder_dim, encoder_dim, bias=False).to(device)

# forward 
q_tensor = q_value.forward(inputs)
k_tensor = k_value.forward(inputs)
v_tensor = v_value.forward(inputs)


concat_attention = torch.cat((q_tensor, k_tensor, v_tensor), dim=0)
print(concat_attention.size())








# Attention mechanism 


# hidden layers encoder
#h_layer_encoder_1 = torch.FloatTensor([40])
#h_layer_encoder_2 = torch.FloatTensor([78])
#h_layer_encoder_3 = torch.FloatTensor([23])
#h_layer_encoder_4 = torch.FloatTensor([67])

# hidden layer decoder
#h_layer_decoder_1 = torch.FloatTensor([14])


# scalar attention by layer (Dot product technique)
#att_step_1 = torch.dot(h_layer_decoder_1, h_layer_encoder_1)
#att_step_2 = torch.dot(h_layer_decoder_1, h_layer_encoder_2)
#att_step_3 = torch.dot(h_layer_decoder_1, h_layer_encoder_3)
#att_step_4 = torch.dot(h_layer_decoder_1, h_layer_encoder_4)



# attention values (hidden layer encoder multiply first decoder layer) 
#print("\n att 1: " , att_step_1)
#print("\n att 2: " , att_step_2)
#print("\n att 3: " , att_step_3)
#print("\n att 4: " , att_step_4)
#print("\n")


# concat attention results 
#concat_att = torch.stack((att_step_1, att_step_2, att_step_3, att_step_4), dim=0)


# pass softmax function (1D)
#att_softmax = nn.functional.softmax(concat_att, dim=0)


# summatory 
#sum = torch.sum(att_softmax * h_layer_decoder_1, dim=0)



#print("Attention softmax: ", att_softmax)
#print("\n\n Sum: ", sum)

