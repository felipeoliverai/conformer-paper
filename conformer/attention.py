import torch 
import torch.nn as nn 


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model=256, q_value=0, k_value=0, v_value=0):
        super(ScaledDotProductAttention, self).__init__()

        # Scaled Dot-Product Attention (Self-attention)
        self.q_value = q_value
        self.k_value = k_value 
        self.v_value = v_value

    def forward(self, inputs):
        return self.sequential(inputs)



#class PositionalEmbedding(nn.Module)



class MHSA(nn.Module):
    def __init__(self, encoder_dim = 256):
        super(MHSA, self).__init__()


        self.sequential = nn.Sequential(nn.LayerNorm(encoder_dim))

    # Self attention
















# Attention mechanism 


# hidden layers encoder
h_layer_encoder_1 = torch.FloatTensor([40])
h_layer_encoder_2 = torch.FloatTensor([78])
h_layer_encoder_3 = torch.FloatTensor([23])
h_layer_encoder_4 = torch.FloatTensor([67])

# hidden layer decoder
h_layer_decoder_1 = torch.FloatTensor([14])


# scalar attention by layer (Dot product technique)
att_step_1 = torch.dot(h_layer_decoder_1, h_layer_encoder_1)
att_step_2 = torch.dot(h_layer_decoder_1, h_layer_encoder_2)
att_step_3 = torch.dot(h_layer_decoder_1, h_layer_encoder_3)
att_step_4 = torch.dot(h_layer_decoder_1, h_layer_encoder_4)



# attention values (hidden layer encoder multiply first decoder layer) 
print("\n att 1: " , att_step_1)
print("\n att 2: " , att_step_2)
print("\n att 3: " , att_step_3)
print("\n att 4: " , att_step_4)
print("\n")


# concat attention results 
concat_att = torch.stack((att_step_1, att_step_2, att_step_3, att_step_4), dim=0)


# pass softmax function (1D)
att_softmax = nn.functional.softmax(concat_att, dim=0)


# summatory 
sum = torch.sum(att_softmax * h_layer_decoder_1, dim=0)



print("Attention softmax: ", att_softmax)
print("\n\n Sum: ", sum)

