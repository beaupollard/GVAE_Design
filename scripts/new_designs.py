import numpy as np
import random
import math
import matplotlib.pyplot as plt
import utils as util
import copy
import time
import torch
import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import utils
from sim_ctrl_new_API import main_run
import envs_for_beau as env_util

# adding Folder_2 to the system path
sys.path.insert(0, '/home/beau/Documents/GVAE_Design/NN')
from DVAE import VAE

def build_ters():
    if terrain==0:
        final_pos, _, _, b0 = env_util.build_gaussian_field(sim)
    elif terrain==1:
        final_pos, _, _, b0 = env_util.build_steps(sim,25)
    elif terrain==2:
        final_pos, _, b0 = env_util.build_slope(sim,25)
    return final_pos, b0


model=VAE()
model.to("cpu")
model.load_state_dict(torch.load("../NN/current_model_org",map_location=torch.device('cpu')))

d1=torch.load('../NN/data12122023.pt')#smd.run_sim(run_nums=2,out_data=2,num_repeats=1)
prev_data=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)

## Set the environment
# terrains = ['rough','steps','slope']
terrain=2
obj=np.array([[-0.3,-0.3,0.4]]).T
org_reals, org_ints, org_results, bo_reals, bo_ints, best_rec, yres = model.BO(prev_data,num_iters=100,num_samples=100,terrain=terrain,obj=obj)

client = RemoteAPIClient()
sim = client.getObject('sim')
sim.closeScene()
sim_results=[]

## Run the org first
nodes, edges = util.create_vehicles(org_reals[0],org_ints[0])
joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes)

final_pos, b0=build_ters()

success, time_sim, ave_torque, max_torque, ave_power, max_power, pin = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
sim.removeObject(b0)
sim_results.append([success,time_sim,ave_torque,max_torque,ave_power,max_power,pin[0],pin[1],pin[2]])
sim.closeScene()

## Run the bo
nodes, edges = util.create_vehicles(bo_reals[0],bo_ints[0])
joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes)

final_pos, b0=build_ters()

success, time_sim, ave_torque, max_torque, ave_power, max_power, pin = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
sim.removeObject(b0)
sim_results.append([success,time_sim,ave_torque,max_torque,ave_power,max_power,pin[0],pin[1],pin[2]])
sim.closeScene()

# nodes, edges = util.create_vehicles(bo_reals[0],bo_ints[0])
# joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes)

# final_pos, b0=build_ters()

# success, time_sim, ave_torque, max_torque, ave_power, max_power, pin = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
# sim.removeObject(b0)
# sim_results.append([success,time_sim,ave_torque,max_torque,ave_power,max_power,pin[0],pin[1],pin[2]])
# sim.closeScene()

sim_results=np.array(sim_results)
out_results=np.zeros((2,3))
for i in range(2):
    out_results[i,0]=sim_results[i,2]/250.
    out_results[i,1]=sim_results[i,4]/275.
    out_results[i,2]=-sim_results[i,6]/(sim_results[i,1]+0.1)*100/29.

yout=out_results@obj

ysort=np.sort((yres@obj).flatten())
idx=[]
for j in range(2):
    for i in range(len(ysort)-1):
        if ysort[i+1]>yout[j] and ysort[i]<=yout[j]:
            idx.append(i)
            continue
print("ORG is as good as " + str(idx[0]/len(ysort))+"\n")
print("BO is as good as " + str(idx[1]/len(ysort))+"\n")
print("finished")

# # x_reals, x_ints, gt_reals, gt_ints = model.design_grads(prev_data)
# x_reals, x_ints, perf_models, torque = model.best_designs(prev_data)

# client = RemoteAPIClient()
# sim = client.getObject('sim')
# motors=[]
# sim.closeScene()

# sim_results=[]
# x_rec=[]
# edge_rec=[]
# pin_rec=[]
# i_rec=[]
# for i in range(len(x_reals)):
#     nodes, edges = util.create_vehicles(x_reals[i],x_ints[i])
#     # nodes, edges = util.create_vehicles(gt_reals,gt_ints)#x_reals[i],x_ints[i])
#     joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes)
#     # final_pos, _, _, b0=utils.build_steps(sim,25,height[i].item(),slope[i].item())
#     # success, time_sim, ave_torque, max_torque, pin = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
#     # pin_rec.append((pin[0]**2+pin[-1]**2)**0.5)
#     # i_rec.append(i)
#     sim.closeScene()
#     # sim_results.append([success,time_sim,ave_torque,max_torque,pin[0],pin[1],pin[2]])
#     # x_rec.append(copy.copy(x_current))#, edge_current
#     # edge_rec.append(copy.copy(edge_current))    
# # plt.plot(perf_models)
# # plt.plot(i_rec,pin_rec,'.b')
# # plt.show()
# # print("hey")
# FS=12

# plt.rcParams.update({'font.size': 22})
# plt.plot(perf_models/30.,'.b',markersize=FS)
# plt.xlabel('Configuration #')
# plt.ylabel('Ascent Speed (m/s)')
# plt.savefig('TorquevsSpeed.png',bbox_inches="tight")

# FS=15

# plt.rcParams.update({'font.size': FS})
# plt.plot(torque,perf_models/30.,'.b',markersize=15)
# plt.xlabel('Average Torque (Nm)')
# plt.ylabel('Ascent Speed (m/s)')
# plt.savefig('TorquevsSpeed.png',bbox_inches="tight")