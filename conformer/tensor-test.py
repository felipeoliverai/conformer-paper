import torch 

x = torch.tensor([5, 6, 7, 9, 10])
y = torch.Tensor([5, 6, 7, 9, 10])
print("\n\n", x.dtype)
print("\n", y.dtype)


a = torch.FloatTensor([[i for i in range(10)]])
print(a)