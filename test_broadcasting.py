import torch

a = torch.ones((3,4,2))
b = torch.ones((3,2))
c = torch.ones((4,2))
d = torch.arange(0,24).reshape((3,4,2))
print(d)
d_resh = d.reshape((4,3,2))
print(d_resh)

res_2 = torch.add(d_resh, -b)
#print(res_2)
#print(res_2.shape)

res_2 = res_2.reshape((3,4,2))
print(res_2)


