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

def save_results(x_rec,edge_rec,sim_results,num,env_name):
    with open('nodes'+str(num)+'_'+env_name+'.txt', 'w') as convert_file:
        for i in x_rec:
            for j in i:

                convert_file.write(json.dumps(j))
                convert_file.write("\n")
            convert_file.write("\n")
            convert_file.write("\n")

    with open('edges'+str(num)+'_'+env_name+'.txt', 'w') as convert_file:
        for i in edge_rec:
            for j in i:

                convert_file.write(json.dumps(j))
                convert_file.write("\n")
            convert_file.write("\n")
            convert_file.write("\n")

    with open('results'+str(num)+'_'+env_name+'.txt', 'w') as convert_file:
        for i in sim_results:
            convert_file.write(json.dumps(i))
            convert_file.write("\n")

def select_terrain():
    if terrain == 0:
        final_pos, _, _, b0=env_util.build_steps(sim)
    elif terrain == 1:
        final_pos, _, b0 = env_util.build_slope(sim,25)
    elif terrain == 2:
        final_pos, _, _, b0=env_util.build_gaussian_field(sim)
    elif terrain == 3:
        final_pos, _, _, b0=env_util.build_rough_slope(sim)
    else:
        final_pos, _, _, b0=env_util.build_gaussian_field_obs(sim)
    return final_pos, b0

terrain = 2
client = RemoteAPIClient()
sim = client.getObject('sim')
sim.closeScene()

x_rec=[]
edge_rec=[]
sim_results=[]
count=0
count_save=0
seed = 2
con=graph_gens(seed_in=seed)

n_samples=150
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
# mut_des=utils.mutation(copy.deepcopy(pool_nodes),fit_func)
## simulate the initial pool of robots ##
for i, nodes_gen in enumerate(pool_nodes):
    joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes_gen)
    final_pos, b0 = select_terrain()
    success, time_sim, ave_torque, max_torque, ave_power, max_power, pin, ave_vel = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
    torque=ave_torque/250.
    power=ave_power/275.
    vel=ave_vel*100/29
    fit_func[i]=(vel*f_obj[0]-torque*f_obj[1]-power*f_obj[2])
    sim_results.append([success,time_sim,ave_torque,max_torque,ave_power,max_power,pin[0],pin[1],pin[2],ave_vel,i])
    x_rec.append(copy.copy(x_current))#, edge_current
    edge_rec.append(copy.copy(edge_current))   
    sim.removeObject(b0)
    sim.closeScene()
# for jj in range(7500):
mean_fit=[]
counter=0
while counter<1000:
    fit_rec=[]

    ## Generate crossover designs ##
    new_des=utils.crossover(copy.copy(pool_nodes))
    ## Mutate the pool_nodes ##
    mut_des=utils.mutation(copy.deepcopy(pool_nodes),fit_func)

    for i, nodes in enumerate(new_des+mut_des):
        joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes)
        final_pos, b0 = select_terrain()
        success, time_sim, ave_torque, max_torque, ave_power, max_power, pin, ave_vel  = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
        torque=ave_torque/250.
        power=ave_power/275.
        vel=ave_vel*100/29
        fit_rec.append(vel*f_obj[0]-torque*f_obj[1]-power*f_obj[2])
        sim.removeObject(b0)
        sim.closeScene()
        sim_results.append([success,time_sim,ave_torque,max_torque,ave_power,max_power,pin[0],pin[1],pin[2],ave_vel,i])
        x_rec.append(copy.copy(x_current))#, edge_current
        edge_rec.append(copy.copy(edge_current))        
        counter+=1
        print(counter)


    save_results(x_rec,edge_rec,sim_results,21,'wall_passage_seed_'+str(seed)+'terrain_'+str(terrain))
    ## Select best samples ##
    total_fit=np.concatenate((fit_func,np.array(fit_rec)))
    fit_index=np.argsort(total_fit)[::-1]
    total_nodes=copy.copy(pool_nodes+new_des+mut_des)
    for i in range(len(pool_nodes)):
        pool_nodes[i]=copy.copy(total_nodes[fit_index[i]])
        fit_func[i]=copy.copy(total_fit[fit_index[i]])
    mean_fit.append(np.mean(fit_func))
    if len(mean_fit)>2:
        if mean_fit[-2]==mean_fit[-1]:
            break
    print(mean_fit[-1])