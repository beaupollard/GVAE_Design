from torch import nn
import torch.nn.functional as F
import torch
from CNN_terrains import CNN
import matplotlib.pyplot as plt
import numpy as np
import math

model = CNN(lr=2e-3)
d1=torch.load('terrain_npy/training_data')
test=torch.utils.data.DataLoader(d1,batch_size=128, shuffle=False)

model.load_state_dict(torch.load('models/terrain2'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for batch in test:
    with torch.no_grad():
        i = batch.to(device)
        model.optimizer.zero_grad()
        _, mu_train, std = model.forward(i.unsqueeze(1))
        x_hat_train=model.decoder(mu_train)



rough = np.load('terrain_npy/rough_slope.npy')
ru=torch.tensor(rough,dtype=torch.float)
ru2=ru.to(device)
_, mu, std = model.forward(ru2.unsqueeze(0))
x_hat=model.decoder(mu)
ximg_rec=(x_hat.reshape((480,640)).to("cpu")).detach().numpy()
ximg=(i[0].reshape((480,640)).to("cpu")).detach().numpy()
plt.imshow(ximg_rec)

np.save('terrain_npy/rough_slope_ze.npy',(mu.to("cpu")).detach().numpy())
np.save('terrain_npy/rough_ze.npy',(mu_train[2].to("cpu")).detach().numpy())
np.save('terrain_npy/slope_ze.npy',(mu_train[0].to("cpu")).detach().numpy())
np.save('terrain_npy/stairs_ze.npy',(mu_train[1].to("cpu")).detach().numpy())
