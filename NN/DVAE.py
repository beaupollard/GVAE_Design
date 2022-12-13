from torch import nn
import torch.nn.functional as F
import torch
from numpy.linalg import eig
import numpy as np

class VAE(nn.Module):
    def __init__(self, enc_out_dim=68, latent_dim=3, input_height=68,lr=3e-3,hidden_layers=64):
        super(VAE, self).__init__()
        self.lr=lr
        self.count=0
        self.kl_weight=0.01
        self.flatten = nn.Flatten()
        self.latent_dim=latent_dim
        self.body_num=4
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_height, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU()            
        )
        self.linear_mu = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            # nn.Tanh()
        )
        self.linear_logstd = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            # nn.Tanh()
        )

        self.decoder_to_ints = nn.Linear(latent_dim, hidden_layers)
        self.decoder_rnn = nn.Linear(hidden_layers, hidden_layers)
        # self.decoder_props_hidden = nn.RNN(input_size=latent_dim, hidden_size=hidden_layers,batch_first=False)

        self.decoder_reals = nn.Sequential(
            nn.Linear(latent_dim, hidden_layers),
            nn.Tanh(),#nn.ReLU(),
            nn.Linear(hidden_layers, 40),
        )
        self.decoder_body_id= nn.Sequential(
            nn.Linear(hidden_layers, 3),
            nn.Softmax()
        )
        self.decoder_joint_id0= nn.Sequential(
            nn.Linear(hidden_layers, 3),
            nn.Softmax()
        )
        self.decoder_prop_id0= nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_layers, 4),
            nn.Softmax()
        ) 
        self.decoder_joint_id1= nn.Sequential(
            nn.Linear(hidden_layers, 3),
            nn.Softmax()
        )
        self.decoder_prop_id1= nn.Sequential(
            nn.Linear(hidden_layers, 4),
            nn.Softmax()
        )  
        self.decoder_joint_id2= nn.Sequential(
            nn.Linear(hidden_layers, 3),
            nn.Softmax()
        )
        self.decoder_prop_id2= nn.Sequential(
            nn.Linear(hidden_layers, 4),
            nn.Softmax()
        )  
        self.decoder_prop_id3= nn.Sequential(
            nn.Linear(hidden_layers, 4),
            nn.Softmax()
        )         
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.optimizer=self.configure_optimizers(lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

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
        zint=self.decoder_to_ints(z)
        body_id = self.decoder_body_id(zint)
        # hidden_prop=torch.zeros_like(zint)
        # hn=zint+self.decoder_rnn(hidden_prop)
        # prop_id=self.decoder_prop_id0(hn)
        # for i in range(3):
        #     hn=zint+self.decoder_rnn(hidden_prop)
        #     prop_id=torch.cat((prop_id,self.decoder_prop_id0(hn)),dim=1)


        prop_id = torch.cat((self.decoder_prop_id0(zint),self.decoder_prop_id1(zint),self.decoder_prop_id2(zint),self.decoder_prop_id3(zint)),dim=1)
        joint_id = torch.cat((self.decoder_joint_id0(zint),self.decoder_joint_id1(zint),self.decoder_joint_id2(zint)),dim=1)

        return xhat, torch.cat((body_id,prop_id,joint_id),dim=1), body_id, prop_id, joint_id

    def ints_loss(self,inp,x_ints):
        loss2=F.cross_entropy(x_ints[:,:3],inp[:,:3],size_average=False)
        for j in range(4):
            loss2+=F.cross_entropy(x_ints[:,3+j*4:3+(j+1)*4],inp[:,3+j*4:3+(j+1)*4],size_average=False)
        # loss=torch.tensor(0,dtype=float)
        # for i, ground_truth in enumerate(inp):
        #     loss+=F.cross_entropy(torch.reshape(x_ints[i,:3],(1,3)),torch.reshape(ground_truth[:3],(1,3)))
        #     for j in range(torch.argmax(x_ints[i,:3])+2):
        #         loss+=F.cross_entropy(torch.reshape(x_ints[i,3+j*4:3+(j+1)*4],(1,4)),torch.reshape(ground_truth[3+j*4:3+(j+1)*4],(1,4)))
        return loss2
            # print('hey')
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

    # def int_loss(self,body_id,prop_id,joint_id,i):
    #     self.lr_ints=1.
    #     recon_loss_ints = self.lr_ints*F.cross_entropy(body_id,i[:,:3])
    #     for j in range(len(i)):
    #         for jj in range(torch.argmax(i[j][:3]).item()+2):
    #             recon_loss_ints+=self.lr_ints*F.cross_entropy(prop_id[j,jj*4:(jj+1)*4],i[j,3+jj*4:3+(jj+1)*4])
    #             if jj < torch.argmax(i[j][:3]).item()+2-1:
    #                 recon_loss_ints+=self.lr_ints*F.cross_entropy(joint_id[j,jj*3:(jj+1)*3],i[j,19+jj*3:19+(jj+1)*3])
    #     return recon_loss_ints

    def training_step(self, batch):
        running_loss=[0.,0.,0.]
        # if self.count==1000:
        #     self.lr=self.lr/5
        #     self.configure_optimizers(lr=self.lr)
        #     self.count=0
        for i in iter(batch):
            self.optimizer.zero_grad()
            
            # encode x to get the mu and variance parameters
            _, mu, std = self.forward(i)

            q=torch.distributions.Normal(mu,std)
            z=q.rsample()

            # decoded
            x_reals, x_ints, body_id, prop_id, joint_id= self.decoder(z)
            recon_loss_ints=self.ints_loss(i[:,40:],x_ints)
            recon_loss_reals = 0.01*F.mse_loss(x_reals,i[:,:40])
            # recon_loss_ints = 1.*F.binary_cross_entropy_with_logits(x_ints,i[:,40:])
            # recon_loss_ints = 500.*F.cross_entropy(x_ints,i[:,40:])
            # recon_loss = self.gaussian_likelihood(torch.cat((x_reals,x_ints),dim=1), self.log_scale, i[0])#F.mse_loss(z,zhat)-F.mse_loss(x_hat,x)#
            kl = (self.kl_divergence(z, mu, std)*self.kl_weight).mean()
            # recon_loss_ints=self.int_loss(body_id,prop_id,joint_id,i[:,40:])
            elbo=(kl+recon_loss_reals+recon_loss_ints)

            elbo.backward()

            self.optimizer.step()
            running_loss[0] += recon_loss_reals.mean().item()
            running_loss[1] += recon_loss_ints.mean().item()
            running_loss[2] += kl.mean().item()#F.mse_loss(zout,z).item()
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
            correct_bodies=np.zeros((6))
            miss_identification_bodies=np.zeros((3,3))
            miss_identification_props=np.zeros((4,4))
            running_loss=[0.,0.,0.]
            for i in iter(batch):
                self.optimizer.zero_grad()
                _, mu, std = self.forward(i)

                q=torch.distributions.Normal(mu,std)
                z=q.rsample()

                # decoded
                x_reals, x_ints, _, _, _= self.decoder(z)
                i_ints=i[:,40:]
                # F.cross_entropy(x_ints[:,:3],i_ints[:,:3])+F.cross_entropy(x_ints[:,3:7],i_ints[:,3:7])+F.cross_entropy(x_ints[:,7:11],i_ints[:,7:11])+F.cross_entropy(x_ints[:,11:15],i_ints[:,11:15])+F.cross_entropy(x_ints[:,15:19],i_ints[:,15:19])+F.cross_entropy(x_ints[:,19:22],i_ints[:,19:22])+F.cross_entropy(x_ints[:,22:25],i_ints[:,22:25])+F.cross_entropy(x_ints[:,25:28],i_ints[:,25:28])
                for j in range(len(i)):
                    ## Determine if body number is correct ##
                    if (torch.argmax(x_ints[j][:3])==torch.argmax(i_ints[j,:3])).detach().numpy()==True:
                        correct_bodies[0]+=1
                    else:
                        correct_bodies[1]+=1
                        miss_identification_bodies[torch.argmax(i_ints[j,:3]).item(),torch.argmax(x_ints[j][:3]).item()]+=1
                    for jj in range(torch.argmax(i_ints[j,:3]).item()+2):
                        if (torch.argmax(x_ints[j][3+4*jj:3+4*(jj+1)])==torch.argmax(i_ints[j,3+4*jj:3+4*(jj+1)])).detach().numpy()==True:
                            correct_bodies[2]+=1
                        else:
                            correct_bodies[3]+=1
                            miss_identification_props[torch.argmax(i_ints[j,3+4*jj:3+4*(jj+1)]).item(),torch.argmax(x_ints[j][3+4*jj:3+4*(jj+1)]).item()]+=1
                        if jj < 3:
                            if (torch.argmax(x_ints[j][3+16+3*jj:3+16+3*(jj+1)])==torch.argmax(i_ints[j,3+16+3*jj:3+16+3*(jj+1)])).detach().numpy()==True:
                                correct_bodies[4]+=1
                            else:
                                correct_bodies[5]+=1

                return correct_bodies, miss_identification_bodies, miss_identification_props
        # return x_hat.detach().numpy(), z.detach().numpy(), x.detach().numpy()