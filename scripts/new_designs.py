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
model.load_state_dict(torch.load("../NN/current_model4",map_location=torch.device("cpu")))
d1=torch.load('../NN/data1.pt')
prev_data=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
x_reals, x_ints = model.design_grads(prev_data)

client = RemoteAPIClient()
sim = client.getObject('sim')
motors=[]
sim.closeScene()

sim_results=[]
x_rec=[]
edge_rec=[]
for i in range(len(x_reals)):
    nodes, edges = util.create_vehicles(x_reals[i],x_ints[i])
    joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes)
    final_pos=utils.build_steps(sim)
    success, time_sim, ave_torque, max_torque, pin = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
    sim.closeScene()
    sim_results.append([success,time_sim,ave_torque,max_torque,pin[0],pin[1],pin[2]])
    x_rec.append(copy.copy(x_current))#, edge_current
    edge_rec.append(copy.copy(edge_current))    
print("hey")