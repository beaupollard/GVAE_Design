from isaacgym import gymapi
from isaacgym import gymtorch
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt

class MPPI():
    def __init__(self,sim,gym,num_dof,num_env,T,active_index,weights,tracked_dofs,tracked_root=[],ctrl_weights=0.*np.ones(4),device='cuda:0'):
        self.sim = sim                                                                          # Store the link to the sim
        self.gym = gym                                                                          # Store the link to Gym environment
        self.device = device                                                                    # Device to store tensors
        self.num_env = num_env                                                                  # Number of envs
        self.T = T                                                                              # MPC horizon
        self.num_dof = num_dof                                                                  # Number of dof's per agent
        self.active_index=active_index                                                          # Number of active dof's per agent
        self.act_num_dof = len(active_index)                                                    # Number of dof's per agent
        self.u = torch.zeros((self.num_env,num_dof,1),dtype=torch.float32, device=self.device)        # current action
        self.U = torch.zeros((self.num_env,num_dof,self.T),dtype=torch.float32, device=self.device)   # sequence of actions
        self.U_weighted = torch.zeros((self.num_env,num_dof,self.T),dtype=torch.float32, device=self.device)
        self.delta_U = torch.zeros((self.num_env,num_dof,self.T),dtype=torch.float32, device=self.device)  # sequence of actions
        self.stored_u = []                                                                      # Stored u value to execute
        self.mu_noise = torch.zeros(self.act_num_dof)                                           # Noise mean for perturbing control
        self.nu = 1000.
        self.u_max = 30.
        self.sigma_noise = torch.diag(self.nu*torch.ones(self.act_num_dof))                     # Noise covariance for perturbing control
        self.noise_dist = MultivariateNormal(self.mu_noise, covariance_matrix=self.sigma_noise) # Noise distribution
        self.noise_sigma_inv = torch.inverse(self.sigma_noise)                                  # Inverse of noise covariance
        self.tracked_dofs=tracked_dofs                                                          # Dof's to track with costs
        self.tracked_root=tracked_root                                                          # Root Dof's to track with costs
        self._lambda = 100.
        self.q_hat = torch.zeros((self.num_env,self.T),dtype=torch.float32, device=self.device) 
        
        # self.gym.prepare_sim(self.sim)                                                          
        # self.gym.simulate(self.sim)

        # ## Aquire tensor descriptors #
        self.root_states_desc = gym.acquire_actor_root_state_tensor(sim)
        self.dof_states_desc = gym.acquire_dof_state_tensor(sim)


        # ## Pytorch interop ##
        root_states = gymtorch.wrap_tensor(self.root_states_desc)
        dof_states = gymtorch.wrap_tensor(self.dof_states_desc)

        # ## View information as a vector of envs ##
        self.root_states_vec = root_states.view(num_env,1,13)
        self.dof_states_vec = dof_states.view(num_env,int(dof_states.size()[0]/num_env),2)
        self.rec_state=torch.zeros((self.num_env,int(dof_states.size()[0]/num_env),self.T),dtype=torch.float32, device=self.device)
        self.new_state=torch.zeros((int(dof_states.size()[0]/num_env),self.T),dtype=torch.float32, device=self.device)
        
        ## Store information used to calculate costs ##
        cost_size=2*len(self.tracked_dofs)+len(self.tracked_root)
        self.tracked_states_vec = torch.cat((self.root_states_vec[:,0,tracked_root],self.dof_states_vec[:,tracked_dofs,0],self.dof_states_vec[:,tracked_dofs,1]),1).view(self.num_env,cost_size,1)
        
        self.Q = torch.zeros((self.num_env,cost_size,cost_size),dtype=torch.float32, device=self.device)
        self.R = torch.zeros((self.num_env,self.act_num_dof,self.act_num_dof),dtype=torch.float32, device=self.device)
        for i in range(self.num_env):
            self.Q[i,:,:]=torch.diag(torch.tensor(weights,dtype=torch.float32, device=self.device))
            self.R[i,:,:]=torch.diag(torch.tensor(ctrl_weights,dtype=torch.float32, device=self.device))
        self.running_costs = torch.zeros((num_env,self.act_num_dof),dtype=torch.float32, device=self.device)
        # self.running_weight = torch.zeros(num_env,dtype=torch.float32, device=self.device)

    def _running_costs(self,delta_u,u):
        '''
        Calculate the running costs 
        '''
        q = torch.bmm(torch.bmm(self.tracked_states_vec.mT,self.Q),self.tracked_states_vec).flatten()
        q+=(1-1/self.nu)/2*torch.bmm(torch.bmm(delta_u.mT,self.R),delta_u).flatten()
        q+=torch.bmm(torch.bmm(u.mT,self.R),delta_u).flatten()+1/2*torch.bmm(torch.bmm(u.mT,self.R),u).flatten()
        return q
    
    def _final_costs(self,delta_u,u):
        '''
        Calculate the running costs 
        '''
        q = torch.bmm(torch.bmm(self.tracked_states_vec.mT,10*self.Q),self.tracked_states_vec).flatten()
        # q+=(1-1/self.nu)/2*torch.bmm(torch.bmm(delta_u.mT,self.R),delta_u).flatten()
        # q+=torch.bmm(torch.bmm(u.mT,self.R),delta_u).flatten()+1/2*torch.bmm(torch.bmm(u.mT,self.R),u).flatten()
        return q

    def _clip_inputs(self):
        perturbed_action=torch.clip((self.U+self.delta_U),-self.u_max,self.u_max)
        self.delta_U=self.U-perturbed_action

    def _get_samples(self):
        '''
        Generate random control samples
        '''
        samples = self.noise_dist.sample((self.num_env,self.T)) 
        for i, act_j in enumerate(self.active_index):
            self.delta_U[:,act_j,:] = samples[:,:,i].to(self.device)

    def _rollout(self):
        '''
        Rollout the dynamics with the perturbed control inputs
        '''
        self._get_samples()
        running_weight=torch.zeros(1,dtype=torch.float32, device=self.device)
        self._clip_inputs()
        for i in range(self.T):
            forces_desc = gymtorch.unwrap_tensor((self.U[:,:,i]+self.delta_U[:,:,i]).contiguous())
            self.gym.set_dof_actuation_force_tensor(self.sim,forces_desc)
            self.gym.simulate(self.sim)
            self._refresh_state()
            self.rec_state[:,:,i]=self.dof_states_vec[:,:,0]
            if i < self.T-1:
                self.q_hat[:,i]=torch.exp(-1/self._lambda*self._running_costs(self.delta_U[:,self.active_index,i:i+1],self.U[:,self.active_index,i:i+1]))
                running_weight+=torch.sum(self.q_hat[:,i])
            else:
                self.q_hat[:,i]=torch.exp(-1/self._lambda*self._final_costs(self.delta_U[:,self.active_index,i:i+1],self.U[:,self.active_index,i:i+1]))
                running_weight+=torch.sum(self.q_hat[:,i])
            for j, act_j in enumerate(self.active_index):
                self.U_weighted[:,act_j,i]= self.q_hat[:,i]*self.delta_U[:,act_j,i]
        
        for i in range(self.T-1):
            self.U[:,:,i]+=torch.sum(self.U_weighted[:,:,i],0)/torch.sum(self.q_hat[:,i]+10**(-30))

            # for j in range(self.act_num_dof):
            #     self.running_costs[:,j]+=torch.mul(costs,self.delta_U[:,self.active_index[j],i])
            #     self.U[:,self.active_index[j],i]+=torch.div(self.running_costs[:,j],self.running_weight)
        self.U=torch.clip((self.U),-self.u_max,self.u_max)
        if torch.nonzero(torch.isnan(self.U.flatten())).size()[0]>0:
            print('stop')
    
    def test_U(self):
        for i in range(self.T):
            forces_desc = gymtorch.unwrap_tensor((self.U[:,:,i]).contiguous())
            self.gym.set_dof_actuation_force_tensor(self.sim,forces_desc)
            self.gym.simulate(self.sim)
            self._refresh_state()
            self.new_state[:,i]=self.dof_states_vec[0,:,0]

        for i in range(self.num_env):
            plt.plot(self.rec_state[i,-1,:].cpu(),'b')
        plt.plot(self.new_state[-1,:].cpu(),'r')
        plt.show()

    def get_command(self,current_root_state,current_dof_state):
        '''
        Given a state this function will return the best sequence of commands.
        '''
        self.U = torch.roll(self.U,-1,2)
        self.running_costs[:]=0.
        # self.running_weight[:]=0.
        self.prev_root_state = torch.clone(current_root_state[0,:])
        self.prev_dof_state = torch.clone(current_dof_state[:self.num_dof,:])
        self._rollout()
        self.reset_rollout_sim()
        # self.test_U()
        # self.reset_rollout_sim()
        forces_desc = gymtorch.unwrap_tensor(self.U[:,:,0].contiguous())
        self.gym.set_dof_actuation_force_tensor(self.sim,forces_desc)

    def _refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.tracked_states_vec=torch.cat((self.root_states_vec[:,0,self.tracked_root],self.dof_states_vec[:,self.tracked_dofs,0],self.dof_states_vec[:,self.tracked_dofs,1]),1).view(self.num_env,2*len(self.tracked_dofs)+len(self.tracked_root),1)

    def reset_rollout_sim(self):
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.prev_root_state.repeat(self.num_env).view(self.num_env,13)))
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.prev_dof_state.flatten().repeat(self.num_env).view(self.num_env,self.num_dof,2)))
