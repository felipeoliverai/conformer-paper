from audioop import bias
from turtle import forward
from requests import head
import torch 
import torch.nn as nn 
import torch.nn.functional as F

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

    def forward(self, q_value: torch.tensor, k_value: torch.tensor, v_value = torch.tensor, n_dim=256):
        

        # n_dim to Tensor type 
        n_dim = torch.Tensor(n_dim).to(device) 

        # matmul (q * kT) "t represents a tranpose matrix"
        #q_k_matmul = torch.matmul(q_value, torch.t(k_value))
        q_k_matmul = ((torch.matmul(q_value, torch.t(k_value)) / torch.sqrt(n_dim)) * v_value)

        # softmax 
        #scaled_dot_product = F.softmax(((q_k_matmul / torch.sqrt(n_dim)) * v_value), -1)
        scaled_dot_product = F.softmax(q_k_matmul, -1)


        
        #scaled_dot_product = torch.softmax((torch.matmul(q_value, torch.t(k_value)) / torch.sqrt(n_dim)) * v_value).to(device)

        return scaled_dot_product



#class PositionalEmbedding(nn.Module)




class MHSA(nn.Module):

    def __init__(self, n_head = 5, encoder_dim = 256):
        super(MHSA, self).__init__()

        # number of Heads
        self.n_head = n_head

        # Q,K,V vectors (linear transformations)
        self.fc_q = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.fc_k = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.fc_v = nn.Linear(encoder_dim, encoder_dim, bias=False)

        # scaled dot product layer 
        self.attention = ScaledDotProductAttention(q_value=self.fc_q, k_value=self.fc_k, v_value=self.fc_v)

        # Linear transformation 
        self.linear_transform = nn.Linear(encoder_dim, encoder_dim, bias=False) # fix 


    def forward(self, inputs):

        # linear Q,K,V
        Q = self.fc_q(inputs)
        K = self.fc_k(inputs)
        V = self.fc_v(inputs)

        # list to concat 
        scaled_dot_product_result = []

        # Scaled Dot Product layer
        for _ in range(0, self.n_head):
            scaled_dot_product = self.attention(Q, K, V)
            scaled_dot_product_result.append(scaled_dot_product)
            #multi_heads = torch.cat([scaled_dot_product])
        
        scaled_dot_tensor = torch.stack(scaled_dot_product_result)
        #concat = torch.cat((scaled_dot_tensor))
        
        linear_transform = self.linear_transform(scaled_dot_tensor)

        

        #multi_heads = torch.cat([scaled_dot_product])
        # concat scaled dot product matrix
        #concat_attention = torch.cat([multi_heads], dim=0)

        return linear_transform
        




torch.manual_seed(42)
inputs = torch.rand(256, device=device)
model = MHSA(encoder_dim=256).to(device)
print(model)
output = model.forward(inputs=inputs)
print(output.size())

#q_value = nn.Linear(encoder_dim, encoder_dim, bias=False).to(device)
#k_value = nn.Linear(encoder_dim, encoder_dim, bias=False).to(device)
#v_value = nn.Linear(encoder_dim, encoder_dim, bias=False).to(device)

# forward 
#q_tensor = q_value.forward(inputs)
#k_tensor = k_value.forward(inputs)
#v_tensor = v_value.forward(inputs)


#concat_attention = torch.cat((q_tensor, k_tensor, v_tensor), dim=0)
#print(concat_attention.size())








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

