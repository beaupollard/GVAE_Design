import numpy as np
import random
import math
import matplotlib.pyplot as plt
import utils as util
import copy
import time
import torch
import sys
from zmqRemoteApi import RemoteAPIClient
import utils
from sim_ctrl_new_API import main_run

# adding Folder_2 to the system path
sys.path.insert(0, '/home/beau/Documents/GVAE_Design/NN')
from DVAE import VAE
model=VAE()
model.load_state_dict(torch.load("../NN/current_model_updatedv7",map_location=torch.device("cpu")))
d1=torch.load('../NN/data01192023.pt')
# d1=torch.load('../NN/datatest3.pt')
prev_data=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
# x_reals, x_ints, gt_reals, gt_ints = model.design_grads(prev_data)
x_reals, x_ints, perf_models, height, slope = model.best_designs(prev_data)

client = RemoteAPIClient()
sim = client.getObject('sim')
motors=[]
sim.closeScene()

sim_results=[]
x_rec=[]
edge_rec=[]
pin_rec=[]
i_rec=[]
for i in range(len(x_reals)):
    nodes, edges = util.create_vehicles(x_reals[i],x_ints[i])
    # nodes, edges = util.create_vehicles(gt_reals,gt_ints)#x_reals[i],x_ints[i])
    joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes)
    final_pos, _, _, b0=utils.build_steps(sim,25,height[i].item(),slope[i].item())
    success, time_sim, ave_torque, max_torque, pin = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
    pin_rec.append((pin[0]**2+pin[-1]**2)**0.5)
    i_rec.append(i)
    sim.closeScene()
    sim_results.append([success,time_sim,ave_torque,max_torque,pin[0],pin[1],pin[2]])
    x_rec.append(copy.copy(x_current))#, edge_current
    edge_rec.append(copy.copy(edge_current))    
plt.plot(perf_models)
plt.plot(i_rec,pin_rec,'.b')
plt.show()
print("hey")