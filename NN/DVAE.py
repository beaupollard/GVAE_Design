from torch import nn
import torch.nn.functional as F
import torch
from numpy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import copy
import math

class VAE(nn.Module):
    def __init__(self, enc_out_dim=68, latent_dim=16, input_height=68,lr=2e-3,hidden_layers=64,dec_hidden_layers=128,performance_out=3,env_inputs=128,seed=0,ze_dim=8,cond_p=True):
        super(VAE, self).__init__()
        torch.manual_seed(seed)
        self.reals_weight=1.
        self.ints_weight=1.
        self.kl_weight=0.1
        self.perf_weight=1000.
        self.dec_hidden_layers=dec_hidden_layers
        self.lr=lr
        self.count=0
        self.flatten = nn.Flatten()
        self.latent_dim=latent_dim
        self.body_num=4
        self.q_prior=torch.distributions.Normal(0.,1.)
        self.performance_out=performance_out
        decode_perf=performance_out
        input_height+=ze_dim
        self.cond_p = cond_p
        if self.cond_p == True:
            encoder_input=input_height+performance_out
        else:
            encoder_input=input_height

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(encoder_input, 2*hidden_layers),
            # nn.Linear(input_height+performance_out, 2*hidden_layers),
            nn.Tanh(),
            nn.Linear(2*hidden_layers, 2*hidden_layers),
            nn.Tanh(),
            nn.Linear(2*hidden_layers, hidden_layers),
            # nn.Tanh()         
        )
        self.linear_mu = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            # nn.Tanh()
        )
        self.linear_logstd = nn.Sequential(
            nn.Linear(hidden_layers, latent_dim),
            # nn.ReLU()
        )
        self.pp_mu = nn.Sequential(
            nn.Linear(hidden_layers, performance_out),
            # nn.Tanh()
        )
        self.pp_logstd = nn.Sequential(
            nn.Linear(hidden_layers, performance_out),
            # nn.ReLU()
        )
        self.decoder_to_ints = nn.Linear(latent_dim+decode_perf, dec_hidden_layers)
        self.decoder_rnn = nn.Linear(latent_dim+decode_perf, dec_hidden_layers)
        self.decoder_rnn_hidden = nn.Linear(dec_hidden_layers, dec_hidden_layers)
        # self.decoder_props_hidden = nn.RNN(input_size=latent_dim, hidden_size=hidden_layers,batch_first=False)
        self.forward_pp = nn.Sequential(
            nn.Linear(latent_dim,hidden_layers),
            nn.Tanh(),
            nn.Linear(hidden_layers,hidden_layers),
            nn.Tanh(),
            # nn.Linear(hidden_layers,performance_out)
        )
        self.decoder_reals = nn.Sequential(
            nn.Linear(latent_dim+decode_perf, dec_hidden_layers),
            nn.Tanh(),#nn.ReLU(),
            nn.Linear(dec_hidden_layers, 40),
        )
        self.decoder_body_id= nn.Sequential(
            nn.Linear(dec_hidden_layers, 3),
            nn.Softmax()
        )
        self.decoder_joint_id0= nn.Sequential(
            nn.Linear(dec_hidden_layers, 3),
            nn.Softmax()
        )
        self.decoder_prop_id0= nn.Sequential(
            nn.Tanh(),
            nn.Linear(dec_hidden_layers, 4),
            nn.Softmax()
        ) 
        self.decoder_joint_id1= nn.Sequential(
            nn.Linear(dec_hidden_layers, 3),
            nn.Softmax()
        )
        self.decoder_prop_id1= nn.Sequential(
            nn.Linear(dec_hidden_layers, 4),
            nn.Softmax()
        )  
        self.decoder_joint_id2= nn.Sequential(
            nn.Linear(dec_hidden_layers, 3),
            nn.Softmax()
        )
        self.decoder_prop_id2= nn.Sequential(
            nn.Linear(dec_hidden_layers, 4),
            nn.Softmax()
        )  
        self.decoder_prop_id3= nn.Sequential(
            nn.Linear(dec_hidden_layers, 4),
            nn.Softmax()
        )
        self.org = nn.Sequential(
            nn.Linear(ze_dim+performance_out, 128),
            nn.Tanh(),#nn.ReLU()
            nn.Linear(128, 128),
            nn.Tanh(),#nn.ReLU(),
            # nn.Linear(128, latent_dim),
        )  
        self.org_mu = nn.Sequential(
            nn.Linear(128, latent_dim),
            # nn.Tanh()
        )
        self.org_logstd = nn.Sequential(
            nn.Linear(128, latent_dim),
            # nn.ReLU()
        )       
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.optimizer=self.configure_optimizers(lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        # self.user_weights=torch(np.array([[-0.05,-0.05,0.9],[-0.1,-0.1,0.8],[-0.15,-0.15,0.7],[-0.2,-0.2,0.6],[-0.25,-0.25,0.5],[-0.3,-0.3,0.4],[-0.35,-0.35,0.3]]),dtype=torch.float)

    def performance_predict(self,x):
        logits = self.forward_pp(x)
        mu = self.pp_mu(logits)
        logstd = torch.exp(self.pp_logstd(logits)/2)
        # z = self.reparametrize(mu,logstd)
        
        return mu, logstd
    
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

    def org_forward(self,x):
        logits = self.org(x)
        mu = self.org_mu(logits)
        logstd = torch.exp(self.org_logstd(logits)/2)
        return mu, logstd

    def decoder(self,z):
        h0=torch.zeros(self.dec_hidden_layers).to(z.device)
        xhat = self.decoder_reals(z)
        z0=self.decoder_rnn(z)
        h0=z0+self.decoder_rnn_hidden(h0)
        body_id = self.decoder_body_id(h0)
        h0=z0+self.decoder_rnn_hidden(h0)
        prop_id = self.decoder_prop_id0(h0)
        joint_id = self.decoder_joint_id0(h0)
        for i in range(3):
            h0=z0+self.decoder_rnn_hidden(h0)
            prop_id=torch.cat((prop_id,self.decoder_prop_id0(h0)),dim=1)
            if i<2:
                joint_id=torch.cat((joint_id,self.decoder_joint_id0(h0)),dim=1)

        return xhat, torch.cat((body_id,prop_id,joint_id),dim=1), body_id, prop_id, joint_id

    def ints_loss(self,inp,x_ints):
        loss2=F.cross_entropy(x_ints[:,:3],inp[:,:3],size_average=False)
        for j in range(4):
            loss2+=F.cross_entropy(x_ints[:,3+j*4:3+(j+1)*4],inp[:,3+j*4:3+(j+1)*4],size_average=False)
            if j<3:
                loss2+=F.cross_entropy(x_ints[:,19+j*3:19+(j+1)*3],inp[:,19+j*3:19+(j+1)*3],size_average=False)
        return loss2

    def configure_optimizers(self,lr=1e-4):
        return torch.optim.AdamW(self.parameters(), lr=lr)

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

    def training_step(self, batch,device):
        running_loss=[0.,0.,0.,0.,0.]
        
        for i in iter(batch):
            self.optimizer.zero_grad()
       
            x = i[0].to(device)
            y = i[1].to(device)
            # user_ind=np.random.randint(0, len(self.user_weights), size=len(x))
            
            rand_weights=torch.rand((len(x),3),dtype=torch.float,device=device)
            rand_weights[:,-1]+=1.
            user_weights=torch.div(rand_weights.T,torch.sum(rand_weights,dim=1)).T
            user_weights[:,:-1]*=-1.                 
            # encode x to get the mu and variance parameters
            if self.cond_p == True:
                _, mu, std = self.forward(torch.cat((x,y),dim=1))
            else:
                _, mu, std = self.forward(x)   

            q=torch.distributions.Normal(mu,std)
            z=q.rsample()

            # decoded
            # pp_z_mu, pp_z_logstd=self.performance_predict(torch.cat((z,x[:,68:]),dim=1))
            pp_z_mu, pp_z_logstd=self.performance_predict(z)
            # performance_dist=torch.distributions.Normal(pp_z_mu, pp_z_logstd)
            performance_est=pp_z_mu#performance_dist.rsample()

            x_reals, x_ints, body_id, prop_id, joint_id= self.decoder(torch.cat((z,performance_est),dim=1))
            recon_loss_ints=self.ints_weight*self.ints_loss(x[:,40:68],x_ints)
            recon_loss_reals = self.reals_weight*F.mse_loss(x_reals,x[:,:40])


            # performance_est=self.performance_predict(torch.cat((z,x[:,68:]),axis=1))
            recon_perf = self.perf_weight*F.mse_loss(performance_est,y)

            zorg, orgstd = self.org_forward(torch.cat((user_weights,x[:,68:]),dim=1))
            qorg=torch.distributions.Normal(zorg, orgstd)
            # zorg = self.org(torch.cat((user_weights,x[:,68:]),dim=1))
            # Predict performance
            # pp_org_mu, pp_org_logstd=self.performance_predict(z)
            # performance_org_dist=torch.distributions.Normal(pp_org_mu, pp_org_logstd)     
            # performance_org=performance_org_dist.rsample()       
            # performance_org=self.performance_predict(qorg.rsample())

            ## find robots that did best here


            ## just try to maximize the reward function
            # results=-0.1*torch.sum(torch.mul(performance_org,user_weights),dim=1)
            # results=(torch.distributions.kl.kl_divergence(qorg, self.q_prior).sum(1).mean())*self.kl_weight
            # recon_loss_ints = 1.*F.binary_cross_entropy_with_logits(x_ints,i[:,40:])
            # recon_loss_ints = 500.*F.cross_entropy(x_ints,i[:,40:])
            # recon_loss = self.gaussian_likelihood(torch.cat((x_reals,x_ints),dim=1), self.log_scale, i[0])#F.mse_loss(z,zhat)-F.mse_loss(x_hat,x)#
            #kl = (self.kl_divergence(z, mu, std)*self.kl_weight).mean()
            # kl = (torch.distributions.kl.kl_divergence(q, self.q_prior).sum(1).mean()+torch.distributions.kl.kl_divergence(performance_dist, torch.distributions.Normal(0.,0.1)).sum(1).mean())*self.kl_weight
            kl = (torch.distributions.kl.kl_divergence(q, self.q_prior).sum(1).mean())*self.kl_weight
            # recon_loss_ints=self.int_loss(body_id,prop_id,joint_id,i[:,40:])
            elbo=(kl+recon_loss_reals+recon_loss_ints+recon_perf)

            elbo.backward()

            self.optimizer.step()
            running_loss[0] += recon_loss_reals.mean().item()
            running_loss[1] += recon_loss_ints.mean().item()
            running_loss[2] += kl.mean().item()#F.mse_loss(zout,z).item()
            running_loss[3] += recon_perf.mean().item()
            # running_loss[4] += results.mean().item()
            # running_loss[2] += lin_loss.item()
            # lin_ap.append(lin_loss.item())
        self.count+=1
        return running_loss

    def test_val(self,batch,device):
        with torch.no_grad():
            running_loss=[0.,0.,0.,0.,0.]

            for i in iter(batch):

                self.optimizer.zero_grad()
        
                x = i[0].to(device)
                y = i[1].to(device)
                
                rand_weights=torch.rand((len(x),3),dtype=torch.float,device=device)
                rand_weights[:,-1]+=1.
                user_weights=torch.div(rand_weights.T,torch.sum(rand_weights,dim=1)).T
                user_weights[:,:-1]*=-1.                 
                # encode x to get the mu and variance parameters
                if self.cond_p == True:
                    _, mu, std = self.forward(torch.cat((x,y),dim=1))
                else:
                    _, mu, std = self.forward(x)   

                q=torch.distributions.Normal(mu,std)
                z=q.rsample()

                # decoded
                # pp_z_mu, pp_z_logstd=self.performance_predict(torch.cat((z,x[:,68:]),dim=1))
                pp_z_mu, pp_z_logstd=self.performance_predict(z)
                # performance_dist=torch.distributions.Normal(pp_z_mu, pp_z_logstd)
                performance_est=pp_z_mu#performance_dist.rsample()

                x_reals, x_ints, body_id, prop_id, joint_id= self.decoder(torch.cat((z,performance_est),dim=1))
                recon_loss_ints=self.ints_weight*self.ints_loss(x[:,40:68],x_ints)
                recon_loss_reals = self.reals_weight*F.mse_loss(x_reals,x[:,:40])

                # performance_est=self.performance_predict(torch.cat((z,x[:,68:]),axis=1))
                recon_perf = self.perf_weight*F.mse_loss(performance_est,y)

                zorg, orgstd = self.org_forward(torch.cat((user_weights,x[:,68:]),dim=1))
                qorg=torch.distributions.Normal(zorg, orgstd)

                # Predict performance
                # pp_org_mu, pp_org_logstd=self.performance_predict(z)
                # performance_org_dist=torch.distributions.Normal(pp_org_mu, pp_org_logstd)     
                # performance_org=performance_org_dist.rsample()       

                # kl = (torch.distributions.kl.kl_divergence(q, self.q_prior).sum(1).mean()+torch.distributions.kl.kl_divergence(performance_dist, torch.distributions.Normal(0.,0.1)).sum(1).mean())*self.kl_weight
                kl = (torch.distributions.kl.kl_divergence(q, self.q_prior).sum(1).mean())*self.kl_weight           
                running_loss[0] += recon_loss_reals.mean().item()
                running_loss[1] += recon_loss_ints.mean().item()
                running_loss[2] += kl.mean().item()#F.mse_loss(zout,z).item()
                running_loss[3] += recon_perf.mean().item()
            self.count+=1
        return running_loss
    
    def train_org(self,batch,device):
        res_out=0
        for i in iter(batch):
            self.optimizer.zero_grad()
            x = i[0].to(device)
            y = i[1].to(device)
            BS=2*1024
            user_weights=torch.rand((BS,3),dtype=torch.float,device=device)
            user_weights[:,-1]+=1.0
            for ii in range(len(user_weights)):
                user_weights[ii,:]=user_weights[ii,:]/torch.sum(user_weights[ii,:])
                user_weights[ii,:-1]*=-1.

            with torch.no_grad():
                if self.cond_p == True:
                    _, mu, std = self.forward(torch.cat((x,y),dim=1))
                else:
                    _, mu, std = self.forward(x)   
            
            index_rec=[]
            for ii in range(BS):
                # res=y@user_weights[ii]
                index_rec.append(torch.argmax(y@user_weights[ii]).item())#torch.argmax(torch.sum(res,dim=1)).item())
            
            zorg, orgstd = self.org_forward(torch.cat((10*user_weights,x[index_rec,68:]),dim=1))
            q=torch.distributions.Normal(zorg, orgstd)
            z=q.rsample()
            kl = torch.distributions.kl.kl_divergence(q, torch.distributions.Normal(mu[index_rec],0.5*torch.ones(len(index_rec),16))).sum(1).mean()*self.kl_weight
            # kl = torch.distributions.kl.kl_divergence(q, torch.distributions.Normal(0.,1.)).sum(1).mean()*self.kl_weight
            mse_L=F.mse_loss(mu[index_rec],zorg)*10
            # kl_loss=
            loss=mse_L+kl*0.01
            loss.backward()

            self.optimizer.step()            
            # qorg=torch.distributions.Normal(zorg, orgstd)


        #     zorg, orgstd = self.org_forward(torch.cat((user_weights,x[:,68:]),dim=1))
        #     qorg=torch.distributions.Normal(zorg, orgstd)
        #     # zorg = self.org(torch.cat((user_weights,x[:,68:]),dim=1))
        #     # Predict performance
        #     performance_org=self.performance_predict(qorg.rsample())            
        #     # Generate random weights
        #     user_weights=torch.rand((len(x),3),dtype=torch.float,device=device)
        #     user_weights[:,:4]*=-.01
        #     # user_weights=torch.zeros((len(x),5),dtype=torch.float,device=device)
        #     # user_weights[:,-1]=1.            
        #     # user_weights=user_weights/torch.sum(user_weights)

        #     z = self.org(torch.cat((user_weights,x[:,68:]),dim=1))
            
        #     # Predict performance
        #     with torch.no_grad():
        #         performance_est=self.performance_predict(torch.cat((z,x[:,68:]),axis=1))

        #     ## find robots that did best here


        #     ## just try to maximize the reward function
        #     results=-torch.sum(torch.mul(performance_est,user_weights),dim=1)
        #     results.mean().backward()
        #     self.optimizer.step()
        #     res_out+=results.mean().item()
        # return res_out
        return mse_L.item(), kl.item()

    # def train_org(self,batch,device):
    #     res_out=0
    #     for i in iter(batch):
    #         self.optimizer.zero_grad()
    #         x = i[0].to(device)
    #         y = i[1].to(device)
    #         BS=10*1024
    #         weights=torch.tensor([-0.3,-0.3,0.4],dtype=torch.float)
    #         user_weights = torch.tile(weights, (len(y), 1))
    #         # user_weights=torch.rand((BS,3),dtype=torch.float,device=device)
    #         # for ii in range(len(user_weights)):
    #         #     user_weights[ii,:]=user_weights[ii,:]/torch.sum(user_weights[ii,:])
    #         #     user_weights[ii,:-1]*=-1.

    #         with torch.no_grad():
    #             _, mu, std = self.forward(x)
            
    #         # index_rec=[]
    #         # for ii in range(BS):
    #         #     res=y*user_weights[ii]
    #         #     index_rec.append(torch.argmax(torch.sum(res,dim=1)).item())
    #         res=y@weights
    #         index_rec=[torch.argmax(res).item()]
    #         zorg, orgstd = self.org_forward(torch.cat((10*user_weights,x[:,68:]),dim=1))
    #         q=torch.distributions.Normal(zorg, orgstd)
    #         z=q.rsample()
    #         kl = torch.distributions.kl.kl_divergence(q, torch.distributions.Normal(mu[index_rec],0.5*torch.ones(len(index_rec),16))).sum(1).mean()*self.kl_weight
    #         # kl = torch.distributions.kl.kl_divergence(q, torch.distributions.Normal(0.,1.)).sum(1).mean()*self.kl_weight
    #         # mse_L=F.mse_loss(mu[index_rec],zorg)*10
    #         mse_L=F.mse_loss(torch.tile(mu[index_rec], (len(y), 1)),zorg)*10
    #         # kl_loss=
    #         loss=mse_L+kl*0.01
    #         loss.backward()

    #         self.optimizer.step()            

    #     return mse_L.item(), kl.item()

    def test(self, batch,device):
        with torch.no_grad():
            correct_bodies=np.zeros((6))
            miss_identification_bodies=np.zeros((3,3))
            miss_identification_props=np.zeros((4,4))
            miss_identification_joints=np.zeros((3,3))
            running_loss=[0.,0.,0.]
            for ii in iter(batch):
                i=ii[0].to(device)
                y=ii[1].to(device)
                self.optimizer.zero_grad()
                if self.cond_p == True:
                    _, mu, std = self.forward(torch.cat((i,y),dim=1))
                else:
                    _, mu, std = self.forward(i)   

                q=torch.distributions.Normal(mu,std)
                z=q.rsample()

                # decoded
                # pp_z_mu, pp_z_logstd=self.performance_predict(torch.cat((z,i[:,68:]),dim=1))
                pp_z_mu, pp_z_logstd=self.performance_predict(z)
                performance_dist=torch.distributions.Normal(pp_z_mu, pp_z_logstd)
                performance_est=pp_z_mu#performance_dist.rsample()

                x_reals, x_ints, _, _, _= self.decoder(torch.cat((mu,performance_est),dim=1))                
                # x_reals, x_ints, _, _, _= self.decoder(mu)
                i_ints=i[:,40:]
                # F.cross_entropy(x_ints[:,:3],i_ints[:,:3])+F.cross_entropy(x_ints[:,3:7],i_ints[:,3:7])+F.cross_entropy(x_ints[:,7:11],i_ints[:,7:11])+F.cross_entropy(x_ints[:,11:15],i_ints[:,11:15])+F.cross_entropy(x_ints[:,15:19],i_ints[:,15:19])+F.cross_entropy(x_ints[:,19:22],i_ints[:,19:22])+F.cross_entropy(x_ints[:,22:25],i_ints[:,22:25])+F.cross_entropy(x_ints[:,25:28],i_ints[:,25:28])
                for j in range(len(i)):
                    ## Determine if body number is correct ##
                    if (torch.argmax(x_ints[j][:3])==torch.argmax(i_ints[j,:3]))==True:
                        correct_bodies[0]+=1
                    else:
                        correct_bodies[1]+=1
                        miss_identification_bodies[torch.argmax(i_ints[j,:3]).item(),torch.argmax(x_ints[j][:3]).item()]+=1
                    for jj in range(torch.argmax(i_ints[j,:3]).item()+2):
                        if (torch.argmax(x_ints[j][3+4*jj:3+4*(jj+1)])==torch.argmax(i_ints[j,3+4*jj:3+4*(jj+1)]))==True:
                            correct_bodies[2]+=1
                        else:
                            correct_bodies[3]+=1
                            miss_identification_props[torch.argmax(i_ints[j,3+4*jj:3+4*(jj+1)]).item(),torch.argmax(x_ints[j][3+4*jj:3+4*(jj+1)]).item()]+=1
                        if jj < 3:
                            if (torch.argmax(x_ints[j][3+16+3*jj:3+16+3*(jj+1)])==torch.argmax(i_ints[j,3+16+3*jj:3+16+3*(jj+1)]))==True:
                                correct_bodies[4]+=1
                            else:
                                correct_bodies[5]+=1
                    # for jj in range(torch.argmax(i_ints[j,:3]).item()+1):
                    #     if (torch.argmax(x_ints[j][19+3*jj:19+3*(jj+1)])==torch.argmax(i_ints[j,19+3*jj:19+3*(jj+1)])).detach().numpy()==True:
                    #         correct_bodies[6]+=1
                    #     else:
                    #         correct_bodies[7]+=1

                return correct_bodies, miss_identification_bodies, miss_identification_props
        # return x_hat.detach().numpy(), z.detach().numpy(), x.detach().numpy()

    def design_grads(self,batch,min_index=0):
        for ii in iter(batch):
            i=ii[0]
            self.optimizer.zero_grad()
            _, mu, std = self.forward(i)
            performance_est=self.performance_predict(mu)
            perf_index=torch.argmin(performance_est[:,min_index])
            z=mu[perf_index]
            x_reals, x_ints, _, _, _= self.decoder(torch.reshape(z,(1,len(z))))
            i_reals = i[perf_index,:40].detach().numpy()
            i_ints = i[perf_index,40:].detach().numpy()
            for i in range(self.latent_dim):
                z[i]=z[i]*(1+0.1)
                reals, ints, _, _, _= self.decoder(torch.reshape(z,(1,len(z))))
                x_reals=torch.cat((x_reals,reals))
                x_ints=torch.cat((x_ints,ints))
                z[i]=z[i]*(1-0.1)
        
        return x_reals.detach().numpy(), x_ints.detach().numpy(), i_reals, i_ints
        # return x_reals.item(), x_ints.item()
    
    def jacobian(self,mu,y,index):
        perf_0=self.performance_predict(torch.cat((mu,y[-2:])))
        dPdz=np.zeros(len(mu))
        eps=0.001
        for i in range(len(mu)):
            mu_add=torch.zeros((len(mu)))
            mu_add[i]=eps

            forward=self.performance_predict(torch.cat((mu+mu_add,y[-2:])))
            mu_add[i]=-eps
            backward=self.performance_predict(torch.cat((mu+mu_add,y[-2:])))            
            dPdz[i]=(forward[0]-backward[0])/(2*eps)
        mu=mu-dPdz
        print('hey')

    def best_designs(self,batch,min_index=-1,num_robots=300):
        for ii in iter(batch):
            i=ii[0]
            y=ii[1]
            self.optimizer.zero_grad()
            _, mu, std = self.forward(i)

            performance_est=self.performance_predict(torch.cat((mu,y[:,-2:]),axis=1))
            travel_dist=((y[:,-5]**2+y[:,-3]**2)**0.5)/30.
            perf_index=torch.topk((y[:,-5]**2+y[:,-3]**2)**0.5,num_robots).indices#torch.argmin(performance_est[:,min_index])
            self.jacobian(mu[perf_index[0]],y[perf_index[0]],2)
            # perf_index=torch.topk(-performance_est[:,min_index],num_robots).indices#torch.argmin(performance_est[:,min_index])
            # z=mu[perf_index]
            # x_reals = i[perf_index,:40]
            # x_ints = i[perf_index,40:]

            # min_dist=0.23
            # index2keep=[]
            # for jj in range(len(travel_dist)):
            #     if travel_dist[jj]>=min_dist:
            #         index2keep.append(y[jj,:].detach().numpy())
            # y=torch.tensor(np.array(index2keep))
            # perf_index=torch.topk(-y[:,2],num_robots).indices
            dist=[]
            for jj in range(len(y)):
                # if i==perf_index[0]:
                #     dist.append(1000)
                # else:
                dist.append(torch.norm(mu[perf_index[0]]-mu[jj]).item())
            perf_index=torch.tensor(np.argsort(np.array(dist)))
            for j in range(num_robots):
                if j==0:
                    z=torch.reshape(mu[perf_index[j]],(1,len(mu[perf_index[j]])))
                    x_reals = torch.reshape(i[perf_index[j],:40],(1,len(i[perf_index[j],:40])))
                    x_ints = torch.reshape(i[perf_index[j],40:],(1,len(i[perf_index[j],40:])))   
                else:               
                    z=torch.cat((z,torch.reshape(mu[perf_index[j]],(1,len(mu[perf_index[j]])))))
                    x_reals = torch.cat((x_reals,torch.reshape(i[perf_index[j],:40],(1,len(i[perf_index[j],:40])))))
                    x_ints = torch.cat((x_ints,torch.reshape(i[perf_index[j],40:],(1,len(i[perf_index[j],40:])))))
        # self.principle_plot(mu,performance_est,perf_index,x_ints,performance_index=0)
        # self.principle_plot(mu,performance_est,perf_index,x_ints,performance_index=0)
        return x_reals.detach().numpy(), x_ints.detach().numpy(), ((y[perf_index[:100].detach().numpy(),-5]**2+y[perf_index[:100].detach().numpy(),-3]**2)**0.5).detach().numpy(), y[perf_index[:100].detach().numpy(),2]

    def principle_plot(self,z,performance_est,highlights,performance_index=0):
        z=StandardScaler().fit_transform(z.detach().numpy())
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(z)

        x0=principalComponents[highlights[0],0]
        y0=principalComponents[highlights[0],1]
        dists=[]
        
        for i in range(len(principalComponents)):
            dists.append((principalComponents[i,0]-x0)**2+(principalComponents[i,1]-y0)**2)
        inds=dists.argsort()
        
        dists=np.array(dists)
        inds=np.array(inds)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(principalComponents[:,0],principalComponents[:,1],performance_est[:,performance_index].detach().numpy())
        ax.scatter3D(principalComponents[highlights,0],principalComponents[highlights,1],performance_est[highlights,performance_index].detach().numpy(),c='k')
        plt.show()      

    def test_org(self,batch,device,user_weight):
        for i in iter(batch):
            self.optimizer.zero_grad()

            # torch.no_grad()
            x = i[0].to(device)
            y = i[1].to(device)
            _, mu, std = self.forward(x)
            user=torch.tile(torch.tensor(user_weight,device=device),(len(y),1))
            actual_results=torch.sum(torch.mul(y,user),dim=1) 
            performance_est=self.performance_predict(mu)

            ## Split up the batch
            split_index=[]
            for i in range(len(x)-1):
                if x[i,-128]!=x[i+1,-128]:
                    split_index.append(i)
            
            y_rough=y[:split_index[0],:]
            x_rough=x[:split_index[0],:]
            mu_rough=mu[:split_index[0],:]

            y_steps=y[split_index[0]+1:split_index[1],:]
            x_steps=x[split_index[0]+1:split_index[1],:]
            mu_steps=mu[split_index[0]+1:split_index[1],:]

            y_slope=y[split_index[1]+1:,:]
            x_slope=x[split_index[1]+1:,:]
            mu_slope=mu[split_index[1]+1:,:]

            sort_ind=2
            sorted_rough, indices_rough = torch.sort(y_rough[:,sort_ind])
            sorted_steps, indices_steps = torch.sort(y_steps[:,sort_ind])
            sorted_slope, indices_slope = torch.sort(y_slope[:,sort_ind])

            mu=[mu_rough.to("cpu"),mu_steps.to("cpu"),mu_slope.to("cpu")]
            x2=[x_rough.to("cpu"),x_steps.to("cpu"),x_slope.to("cpu")]
            y2=[y_rough.to("cpu"),y_steps.to("cpu"),y_slope.to("cpu")]
            sort2=[indices_rough.to("cpu"),indices_steps.to("cpu"),indices_slope.to("cpu")]

            
            index=0
            # for index in range(3):
            user_weights=torch.tensor([-0.1,-0.1,0.8],device=device)
            # zgen = self.org(torch.cat((user_weights,x2[index][0,68:].to(device))))
            zgen, orgstd = self.org_forward(torch.cat((user_weights,x2[index][0,68:].to(device))))
            z=StandardScaler().fit_transform(torch.cat((mu[index],zgen.to("cpu").unsqueeze(dim=0))).detach().numpy())
            # z=StandardScaler().fit_transform(mu[index].detach().numpy())
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(z)

            x0=principalComponents[sort2[index][-10:].detach().numpy(),0]
            y0=principalComponents[sort2[index][-10:].detach().numpy(),1]
            if index==0:
                plt.plot(principalComponents[:,0],principalComponents[:,1],'.b')
                plt.plot(x0,y0,'.r')
                plt.plot(principalComponents[-1,0],principalComponents[-1,1],'xk')
            elif index==1:
                plt.plot(principalComponents[:,0],principalComponents[:,1],'.k')
                plt.plot(x0,y0,'xr')  
                plt.plot(principalComponents[-1,0],principalComponents[-1,1],'xb')
            else:
                plt.plot(principalComponents[:,0],principalComponents[:,1],'.g')
                plt.plot(principalComponents[-1,0],principalComponents[-1,1],'xr')
                plt.plot(x0,y0,'or')                    
            
            ## check the org generated design

            # plt.plot(y[:,-1].to("cpu").detach().numpy())
            # plt.plot(performance_est[:,-1].to("cpu").detach().numpy())
            # self.principle_plot(mu,y,[4],performance_index=0)

    def test_pp(self,batch,device):
        fig, axs = plt.subplots(1)
        count=0
        for i in iter(batch):
            with torch.no_grad():
                x = i[0].to(device)
                y = i[1].to(device)
                _, mu, std = self.forward(x)
                performance_est=self.performance_predict(mu)
                # performance_est=self.performance_predict(torch.cat((mu,x[:,68:]),axis=1))
                sorted, indices = torch.sort(y[:,-1])
                plt.plot(sorted.to("cpu").detach().numpy())
                plt.plot(performance_est[indices,-1].to("cpu").detach().numpy())                
                # axs[count].plot(sorted.to("cpu").detach().numpy())
                # axs[count].plot(performance_est[indices,-1].to("cpu").detach().numpy())
                count+=1

    def split_batch(self,batch):
        for i in iter(batch):

            self.optimizer.zero_grad()

            with torch.no_grad():
                x = i[0].to("cpu")
                y = i[1].to("cpu")
                if self.cond_p == True:
                    _, mu, std = self.forward(torch.cat((x,y),dim=1))
                else:
                    _, mu, std = self.forward(x)            
                ## Split up the batch
                split_index=[]
                for i in range(len(x)-1):
                    if x[i,-1]!=x[i+1,-1]:
                        split_index.append(i)
                
                y2=[y[:split_index[0],:].to("cpu")]
                x2=[x[:split_index[0],:].to("cpu")]
                mu2=[mu[:split_index[0],:].to("cpu")]
                std2=[std[:split_index[0],:].to("cpu")]

                for i in range(len(split_index)-1):
                    y2.append(y[split_index[i]+1:split_index[i+1],:])
                    x2.append(x[split_index[i]+1:split_index[i+1],:])
                    mu2.append(mu[split_index[i]+1:split_index[i+1],:])
                    std2.append(std[split_index[i]+1:split_index[i+1],:])

                y2.append(y[split_index[-1]+1:,:])
                x2.append(x[split_index[-1]+1:,:])
                mu2.append(mu[split_index[-1]+1:,:])
                std2.append(std[split_index[-1]+1:,:])
                # y_rough=y[:split_index[0],:]
                # x_rough=x[:split_index[0],:]
                # mu_rough=mu[:split_index[0],:]

                # y_steps=y[split_index[0]+1:split_index[1],:]
                # x_steps=x[split_index[0]+1:split_index[1],:]
                # mu_steps=mu[split_index[0]+1:split_index[1],:]

                # y_slope=y[split_index[1]+1:,:]
                # x_slope=x[split_index[1]+1:,:]
                # mu_slope=mu[split_index[1]+1:,:]

                # sort_ind=2
                # sorted_rough, indices_rough = torch.sort(y_rough[:,sort_ind])
                # sorted_steps, indices_steps = torch.sort(y_steps[:,sort_ind])
                # sorted_slope, indices_slope = torch.sort(y_slope[:,sort_ind])

                # mu=[mu_rough.to("cpu"),mu_steps.to("cpu"),mu_slope.to("cpu")]
                # x2=[x_rough.to("cpu"),x_steps.to("cpu"),x_slope.to("cpu")]
                # y2=[y_rough.to("cpu"),y_steps.to("cpu"),y_slope.to("cpu")]
                # sort2=[indices_rough.to("cpu"),indices_steps.to("cpu"),indices_slope.to("cpu")]
        
        return mu2, x2, y2, std2

    def BO(self,batch,num_iters=100,num_samples=10,terrain=0,obj=[]):
        mu, x, y, std = self.split_batch(batch) 
        
        z = mu[terrain].detach().numpy()
        xin=x[terrain].detach().numpy()
        yin=y[terrain].detach().numpy()
        stdz=std[terrain].detach().numpy()
        with torch.no_grad():
            move_on=False
            while move_on==False:
                ## Get results from ORG
                user_weights=torch.tensor(obj.flatten(),device="cpu",dtype=torch.float)
                # zgen = self.org(torch.cat((user_weights,x2[index][0,68:].to(device))))
                zgen, orgstd = self.org_forward(torch.cat((user_weights,torch.tensor(xin[0,68:],dtype=torch.float,device="cpu"))))        
                q=torch.distributions.Normal(zgen,orgstd)
                zgen=q.rsample()
                pp_org_mu, pp_org_logstd=self.performance_predict(zgen)
                # performance_org_dist=torch.distributions.Normal(pp_org_mu, pp_org_logstd)   
                org_results = (pp_org_mu@user_weights).detach().numpy()
                # org_results = (self.performance_predict(zgen)@user_weights).detach().numpy()
                org_reals, org_ints, _, _, _= self.decoder(zgen.unsqueeze(0))
                nodes, _ = create_vehicles(org_reals[0],org_ints[0])
                cout=0
                for i in nodes:
                    if i['name']=='prop' and i['radius']<=0.01:
                        cout+=1
                if cout == 0:
                    move_on=True
        index_y=np.argsort((yin@obj).flatten())
        z[index_y[-1]]
        
        ## Get results from BO
        kernel = RBF(length_scale=1.0)
        gp_model = GaussianProcessRegressor(kernel=kernel)
        num_iterations=5
        beta = 1.5    

        zbounds=np.zeros((self.latent_dim,2))
        ## identify the latent ranges
        for i in range(self.latent_dim):
            zbounds[i,0]=0.98*(np.max(z[:,i]).item())
            zbounds[i,1]=0.98*(np.min(z[:,i]).item())
        np.random.seed(0)
        number_of_rows = xin.shape[0] 
        random_indices = np.random.choice(number_of_rows,size=num_samples,replace=False) 
        obj = user_weights.detach().numpy().T

        ## Initial samples
        sample_x = z[random_indices,:]#np.random.choice(z, size=num_samples)
        pp_org_mu, _=self.performance_predict(torch.tensor(sample_x,dtype=torch.float))
        yout = pp_org_mu.detach().numpy()#self.performance_predict(torch.tensor(sample_x,dtype=torch.float)).detach().numpy()
        sample_y = yout@obj
        best_rec=[]#np.zeros((num_iterations,2))
        
        for i in range(num_iters):
            gp_model.fit(sample_x,sample_y)

            ## upper confidence bound
            y_pred, y_std = gp_model.predict(z, return_std=True)

            best_idx = np.argmax(sample_y)
            best_x = sample_x[best_idx]
            best_y = sample_y[best_idx]    
            best_rec.append(copy.copy(best_y.item()))
            # best_rec[i,0]=copy.copy(best_x)
            # best_rec[i,1]=copy.copy(best_y)        
            ucb = y_pred.flatten() + beta * y_std

            if i < num_iters-1:
                sample_x = np.vstack((sample_x,z[np.argmax(ucb)]))
                pp_org_mu, _=self.performance_predict(torch.tensor(sample_x,dtype=torch.float))
                sample_y = pp_org_mu.detach().numpy()@obj
                # sample_y = np.append(sample_y,y[np.argmax(ucb)])
        with torch.no_grad():
            bo_reals, bo_ints, _, _, _= self.decoder(torch.tensor(z[np.argmax(ucb)],dtype=torch.float).unsqueeze(0))
                
        return org_reals, org_ints, org_results, bo_reals, bo_ints, best_rec, yin, z, xin

    def BO_multi(self,batch,terrain=0,obj=[],runs=10,seed=1):
        mu, x, y, _ = self.split_batch(batch) 
        
        z = mu[terrain].detach().numpy()
        xin=x[terrain].detach().numpy()
        yin=y[terrain].detach().numpy()
        real_rec=[]
        int_rec=[]
        org_rec=[]
        torch.manual_seed(seed)
        with torch.no_grad():
            move_on=0
            while move_on<runs:
                ## Get results from ORG
                user_weights=torch.tensor(obj.flatten(),device="cpu",dtype=torch.float)
                # zgen = self.org(torch.cat((user_weights,x2[index][0,68:].to(device))))
                zgen, orgstd = self.org_forward(torch.cat((10*user_weights,torch.tensor(xin[0,68:],dtype=torch.float,device="cpu"))))        
                q=torch.distributions.Normal(zgen,orgstd)
                zgen=q.rsample()
                pp_org_mu, pp_org_logstd=self.performance_predict(zgen)#torch.cat((zgen,x[terrain][0,68:])))
                # pp_org_mu, pp_org_logstd=self.performance_predict(zgen)torch.cat((zgen,x[:,68:]),dim=1)
                # performance_org_dist=torch.distributions.Normal(pp_org_mu, pp_org_logstd)   
                org_results = (pp_org_mu@user_weights).detach().numpy()
                org_reals, org_ints, _, _, _= self.decoder(torch.cat((zgen,pp_org_mu)).unsqueeze(0))
                nodes, _ = create_vehicles(org_reals[0],org_ints[0])
                cout=0
                for i in nodes:
                    if i['name']=='prop' and i['radius']<=0.01:
                        cout+=1
                if cout == 0:
                    real_rec.append(org_reals[0])
                    int_rec.append(org_ints[0])
                    org_rec.append(org_results)
                    move_on+=1
        return real_rec, int_rec, yin, org_rec

def create_vehicles(x_reals,x_ints,num_bodies=4,num_body_reals=3,num_prop_reals=4,num_joint_reals=4,num_prop_ints=4,num_joint_ints=3):
    ## Determine the number of bodies ##
    bodies=np.argmax(x_ints[:3])+2
    body_reals=np.reshape(x_reals[:num_bodies*num_body_reals],(num_bodies,num_body_reals))
    prop_reals=np.reshape(x_reals[num_bodies*num_body_reals:num_bodies*(num_body_reals+num_prop_reals)],(num_bodies,num_prop_reals))
    joint_reals=np.reshape(x_reals[num_bodies*(num_body_reals+num_prop_reals):],(num_bodies-1,num_joint_reals))
    prop_ints=np.reshape(x_ints[3:3+num_prop_ints*num_bodies],(num_bodies,num_prop_ints))
    joint_ints=np.reshape(x_ints[3+num_prop_ints*num_bodies:],(num_bodies-1,num_joint_ints))
    joint_ints=np.vstack((joint_ints,np.array([0,0,1])))
    joint_reals=np.vstack((joint_reals,np.zeros(num_joint_reals)))
    nodes=[]
    edges=[]
    body_id=0
    prop_types=['none','wheel','planet wheel','track']
    for i in range(bodies):
        # self.body_nodes={"name":"body","location":[],"length": 0,"width":20/39.37,"height":12/39.37,"clearance":0,"childern": [],"parents": [],"index":0}
        # self.joint_nodes={"name":"joint","location": [0, 0, 0],"orientation":[0,0,0],"active":[],"childern": [],"parents": [],"index":1}
        # self.prop_nodes={"name":"prop","location": [0, 0, 0],"radius":0,"childern": [],"parents": [],"type":'none'}
        propid=np.argmax(prop_ints[i,:]).item()
        jointid=np.argmax(joint_ints[i,:])
        
        # nodes.append([body_reals[i,0],body_reals[i,1],body_reals[i,1],0.,0.])
        if propid!=0:
            # nodes.append([propid-1, prop_reals[i,0], prop_reals[i,1], prop_reals[i,2], prop_reals[i,3]])
            ## Determine what type of mechanism is next [0=none, 1=wheel, 2=planet wheel, 3=track] ##
            # kinda cheating for right now #
            nodes.append({"name":"body","location":[],"length": body_reals[i,0].item(),"width":0.508001,"height":0.3048006,"clearance":0,"childern": [body_id+1,body_id+2],"parents": [],"index":0})
            nodes.append({"name":"prop","location": [prop_reals[i,1].item(), 0.658001006, -0.102400303],"radius":prop_reals[i,0].item(),"childern": [],"parents": [],"type":prop_types[propid]})
            # This is the actual NN recreation #
            # nodes.append({"name":"body","location":[],"length": body_reals[i,0].item(),"width":body_reals[i,1].item(),"height":body_reals[i,2].item(),"clearance":0,"childern": [body_id+1,body_id+2],"parents": [],"index":0})
            # nodes.append({"name":"prop","location": [prop_reals[i,1].item(), prop_reals[i,2].item(), prop_reals[i,3].item()],"radius":prop_reals[i,0].item(),"childern": [],"parents": [],"type":prop_types[propid]})
            edges.append([body_id,body_id+1])
            edges.append([body_id,body_id+2])
            index_increase=3
        else:
            nodes.append({"name":"body","location":[],"length": body_reals[i,0].item(),"width":body_reals[i,1].item(),"height":body_reals[i,2].item(),"clearance":0,"childern": [body_id+1],"parents": [],"index":0})
            edges.append([body_id,body_id+1])
            index_increase=2
        jori=[0.,0.,0.]
        jori[jointid]=math.sin(math.pi/4)
        nodes.append({"name":"joint","location": [joint_reals[i,2].item(), 0, joint_reals[i,3].item()],"orientation":jori,"active":[joint_reals[i,0].item(),joint_reals[i,1].item()],"childern": [],"parents": [],"index":1})
        # nodes.append([joint_reals[i,0],joint_reals[i,1],jointid,joint_reals[i,2],joint_reals[i,3]])
        body_id+=index_increase
    return nodes, edges

def plot_stds(z,std,mu,stdmu):
    for i in range(len(z)):
        plt.plot(i+1,z[i],'xk')
        plt.plot(i+1,z[i]+3*std[i],'xr')
        plt.plot(i+1,z[i]-3*std[i],'xr')
        plt.plot([i+1, i+1],[z[i]+3*std[i],z[i]-3*std[i]],'-k')
        # plt.plot(i+1,z[i]-3*std[i],'xr')
        plt.plot(i+1,mu[i],'og')
        plt.plot([i+1,i+1],[mu[i]+3*stdmu[i],mu[i]-3*stdmu[i]],'-b')
        # plt.plot(i+1,z[i],'xk')
        # # plt.plot(i+1,z[i]+std[i],'xr')
        # # plt.plot(i+1,z[i]-std[i],'xr')
        # plt.plot(i+1,z[i]+3*std[i],'xr')
        # plt.plot(i+1,z[i]-3*std[i],'xr')
        # plt.plot(i+1,mu[i],'og')