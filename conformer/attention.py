import torch 
import torch.nn as nn 



#class MultiHeadedSelfAttention(nn.Module):

#    def __init__(self):
#        super(MultiHeadedSelfAttention, self).__init__()

#    self.sequential = nn.Sequential(nn.LayerNorm(inputs),
#                                    MHSA(), 
#                                    nn.Dropout())
    

#    def forward(self, inputs):
#        return self.sequential(inputs)




class MHSA(nn.Module):
    def __init__(self):
        super(MHSA, self).__init__()

    # Self attention






# Attention mechanism 

# hidden layers encoder
h_layer_encoder_1 = torch.FloatTensor([40])
h_layer_encoder_2 = torch.FloatTensor([78])
h_layer_encoder_3 = torch.FloatTensor([23])
h_layer_encoder_4 = torch.FloatTensor([67])

# hidden layer decoder
h_layer_decoder_1 = torch.FloatTensor([14])


# scalar attention by layer 
att_step_1 = (h_layer_encoder_1 * h_layer_decoder_1)
att_step_2 = (att_step_1 * h_layer_decoder_1)
att_step_3 = (att_step_2 * h_layer_decoder_1)
att_step_4 = (att_step_3 * h_layer_decoder_1)


# attention values (hidden layer encoder multiply first decoder layer) 
print("\n att 1: " , att_step_1)
print("\n att 2: " , att_step_2)
print("\n att 3: " , att_step_3)
print("\n att 4: " , att_step_4)
print("\n")


# concat attention results 
concat_att = torch.FloatTensor([att_step_1, att_step_2, att_step_3, att_step_4])

# pass softmax function (1D)
att_softmax = nn.functional.softmax(concat_att, dim=0)


# summatory 
sum = torch.sum(att_softmax * h_layer_decoder_1, dim=0)



print("Attention softmax: ", att_softmax)
print("\n\n Sum: ", sum)

