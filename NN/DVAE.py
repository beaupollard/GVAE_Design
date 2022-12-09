from torch import nn
import torch.nn.functional as F
import torch
from numpy.linalg import eig
import numpy as np

class VAE(nn.Module):
    def __init__(self, enc_out_dim=63, latent_dim=10, input_height=63,lr=1e-3,hidden_layers=128):
        super(VAE, self).__init__()
        self.lr=lr
        self.count=0
        self.kl_weight=0.4
        self.flatten = nn.Flatten()
        self.latent_dim=latent_dim
        self.body_num=4
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_height, hidden_layers),
            nn.ReLU(),
        )
        self.linear_mu = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            # nn.Tanh()
        )
        self.linear_logstd = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            # nn.Tanh()
        )

        self.decoder_reals = nn.Sequential(
            nn.Linear(latent_dim, hidden_layers),
            nn.Tanh(),#nn.ReLU(),
            nn.Linear(hidden_layers, 32),
        )
        self.decoder_body_id= nn.Sequential(
            nn.Linear(latent_dim, 3),
            nn.Softmax()
        )
        self.decoder_joint_id= nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.Softmax()
        )
        self.decoder_prop_id= nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.Softmax()
        )        
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.optimizer=self.configure_optimizers(lr=lr)

    def reparametrize(self,mu,logstd):
        if self.training:
            return mu+torch.randn_like(logstd)*torch.exp(logstd)
        else:
            return mu

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        mu = self.linear_mu(logits)
        logstd = torch.exp(self.linear_logstd(logits)/2)
        # z = self.reparametrize(mu,logstd)
        return logits, mu, logstd

    def decoder(self,z):
        xhat = self.decoder_reals(z)
        body_id = self.decoder_body_id(z)
        prop_id = self.decoder_prop_id(z)
        for i in range(self.body_num-1):
            prop_id=torch.cat((prop_id,self.decoder_prop_id(z)),dim=1)
        joint_id = self.decoder_joint_id(z)
        for i in range(self.body_num-2):
            joint_id=torch.cat((joint_id,self.decoder_joint_id(z)),dim=1)        

        return xhat, torch.cat((body_id,prop_id,joint_id),dim=1)


    def configure_optimizers(self,lr=1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
        # return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def training_step(self, batch):
        running_loss=[0.,0.,0.]

        for i in iter(batch):
            self.optimizer.zero_grad()
            
            # encode x to get the mu and variance parameters
            _, mu, std = self.forward(i[0])

            q=torch.distributions.Normal(mu,std)
            z=q.rsample()

            # decoded
            x_reals, x_ints= self.decoder(z)

            recon_loss = self.gaussian_likelihood(torch.cat((x_reals,x_ints),dim=1), self.log_scale, i[0])#F.mse_loss(z,zhat)-F.mse_loss(x_hat,x)#
            kl = self.kl_divergence(z, mu, std)*self.kl_weight
            
            elbo=(kl-recon_loss).mean()

            elbo.backward()

            self.optimizer.step()
            # running_loss[0] += recon_loss.mean().item()
            # running_loss[1] += kl.mean().item()#F.mse_loss(zout,z).item()
            # running_loss[2] += lin_loss.item()
            # lin_ap.append(lin_loss.item())
        self.count+=1
        return running_loss
# import matplotlib.pyplot as plt
# plt.plot(z[:,0].detach().numpy())
# plt.plot(zout[:,0].detach().numpy())
# import matplotlib.pyplot as plt
# plt.plot(x[:,0].detach().numpy())
# plt.plot(x_hat[:,0].detach().numpy())

    def test(self, batch):
        with torch.no_grad():
            
            running_loss=[0.,0.,0.]
            for i in iter(batch):
                self.optimizer.zero_grad()
                x, y = i

                # encode x to get the mu and variance parameters
                x_encoded, mu, std = self.forward(x)

                q=torch.distributions.Normal(mu,std)
                z=q.rsample()

                # decoded
                x_hat, A, B = self.decoder(z)
        return x_hat.detach().numpy(), z.detach().numpy(), x.detach().numpy()