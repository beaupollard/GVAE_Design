from torch import nn
import torch.nn.functional as F
import torch
from CNN_terrains import CNN
import matplotlib.pyplot as plt
import numpy as np
import math

model = CNN(lr=2e-3)
d1=torch.load('terrain_npy/training_data')
test=torch.utils.data.DataLoader(d1,batch_size=256, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
init_kl_weight=0.01
kl_weight=0.01
for epoch in range(1000):
    loss_rec=[0,0]
    for batch in test:
        i = batch.to(device)
        model.optimizer.zero_grad()
        _, mu, std = model.forward(i.unsqueeze(1))
        mu_sample=torch.distributions.Normal(mu, std)
        x_hat=model.decoder(mu_sample.rsample())
        kl = (torch.distributions.kl.kl_divergence(mu_sample, torch.distributions.Normal(0.,1.)).sum(1).mean())*kl_weight
        recon_loss=F.mse_loss(x_hat,i.flatten(1))
        elbo = kl+recon_loss
        elbo.backward()

        model.optimizer.step()
        loss_rec[0] += recon_loss.mean().item()
        loss_rec[1] += kl.mean().item()

    print(epoch, loss_rec)
    if epoch%25 == 0:
        model.scheduler.step()
        kl_weight=init_kl_weight*(1+10*(1-math.exp(-epoch/5000)))


ximg=(i[0].reshape((480,640)).to("cpu")).detach().numpy()
ximg_rec=(x_hat[0].reshape((480,640)).to("cpu")).detach().numpy()
plt.imshow(ximg_rec)
plt.imshow(ximg)

rough = np.load('terrain_npy/rough_slope.npy')
ru=torch.tensor(rough,dtype=torch.float)
ru2=ru.to(device)
_, mu, std = model.forward(ru2.unsqueeze(0))
x_hat=model.decoder(mu)
ximg_rec=(x_hat.reshape((480,640)).to("cpu")).detach().numpy()
plt.imshow(ximg_rec)
