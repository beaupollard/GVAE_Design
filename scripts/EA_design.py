import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
# from zmqRemoteApi import RemoteAPIClient
import utils
import numpy as np
import math
from graph_generator import graph_gens
from sim_ctrl_new_API import main_run
import copy
import numpy as np
import json
from multiprocessing import Process
import envs_for_beau as env_util

def save_results(x_rec,edge_rec,sim_results,num,run_num):
    with open('nodes'+str(num)+'_'+str(run_num)+'.txt', 'w') as convert_file:
        for i in x_rec:
            for j in i:

                convert_file.write(json.dumps(j))
                convert_file.write("\n")
            convert_file.write("\n")
            convert_file.write("\n")

    with open('edges'+str(num)+'_'+str(run_num)+'.txt', 'w') as convert_file:
        for i in edge_rec:
            for j in i:

                convert_file.write(json.dumps(j))
                convert_file.write("\n")
            convert_file.write("\n")
            convert_file.write("\n")

    with open('results'+str(num)+'_'+str(run_num)+'.txt', 'w') as convert_file:
        for i in sim_results:
            convert_file.write(json.dumps(i))
            convert_file.write("\n")

client = RemoteAPIClient()
sim = client.getObject('sim')
sim.closeScene()

x_rec=[]
edge_rec=[]
sim_results=[]
count=0
count_save=0
con=graph_gens(seed_in=1)

n_samples=15
pool_nodes = []
f_obj=np.array([0.4,0.3,0.3])
## Generate the initial population ##
for i in range(n_samples):
    num_props=0
    while num_props<2:
        con.reset()
        num_props, nodes_out=con.generate_concept()
    pool_nodes.append(copy.copy(nodes_out))


fit_func=np.zeros(n_samples)
mut_des=utils.mutation(copy.deepcopy(pool_nodes),fit_func)
## simulate the initial pool of robots ##
for i, nodes_gen in enumerate(pool_nodes):
    joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes_gen)

    # final_pos, _, _, b0=env_util.build_steps(sim)
    # final_pos, _, b0 = env_util.build_slope(sim,25)
    # final_pos, _, _, b0=env_util.build_gaussian_field(sim)field_obs
    # final_pos, _, _, b0=env_util.build_rough_slope(sim)
    final_pos, _, _, b0=env_util.build_gaussian_field_obs(sim)
    success, time_sim, ave_torque, max_torque, ave_power, max_power, pin = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
    torque=ave_torque/250.
    power=ave_power/275.
    vel=-(pin[0]/time_sim)*100/29
    fit_func[i]=(vel*f_obj[0]-torque*f_obj[1]-power*f_obj[2])
    sim.removeObject(b0)
    sim.closeScene()
# for jj in range(7500):
mean_fit=[]
while True:
    fit_rec=[]

    ## Generate crossover designs ##
    new_des=utils.crossover(copy.copy(pool_nodes))
    ## Mutate the pool_nodes ##
    mut_des=utils.mutation(copy.deepcopy(pool_nodes),fit_func)

    for i, nodes in enumerate(new_des+mut_des):
        joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes)

        # final_pos, _, _, b0=env_util.build_steps(sim)
        # final_pos, _, b0 = env_util.build_slope(sim,25)
        # final_pos, _, _, b0=env_util.build_gaussian_field(sim)
        # final_pos, _, _, b0=env_util.build_rough_slope(sim)
        final_pos, _, _, b0=env_util.build_gaussian_field_obs(sim)
        success, time_sim, ave_torque, max_torque, ave_power, max_power, pin = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
        torque=ave_torque/250.
        power=ave_power/275.
        vel=-(pin[0]/time_sim)*100/29
        fit_rec.append(vel*f_obj[0]-torque*f_obj[1]-power*f_obj[2])
        sim.removeObject(b0)
        sim.closeScene()

    ## Select best samples ##
    total_fit=np.concatenate((fit_func,np.array(fit_rec)))
    fit_index=np.argsort(total_fit)[::-1]
    total_nodes=copy.copy(pool_nodes+new_des+mut_des)
    for i in range(len(pool_nodes)):
        pool_nodes[i]=copy.copy(total_nodes[fit_index[i]])
        fit_func[i]=copy.copy(total_fit[fit_index[i]])
    mean_fit.append(np.mean(fit_func))
    print(mean_fit[-1])