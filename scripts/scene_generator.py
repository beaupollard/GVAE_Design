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

def run_multi(ii):
    ## Get ID of open CoppeliaSim scene ##
    ports = [23000,23002,23004,23006]
    client = RemoteAPIClient(port=ports[ii])
    sim = client.getObject('sim')
    # sim = client.require('sim')
    motors=[]
    sim.closeScene()
    # sim.setBoolParam(sim.boolparam_display_enabled,False)
    x_rec=[]
    edge_rec=[]
    sim_results=[]
    client_id=0
    count=0
    count_save=0
    step_height=[6.5/39.39]#[5.5/39.39,6.5/39.39,7.5/39.39]
    slope=[28.0]#[25,32.5,40]
    seed_start=[0,10000,20000,30000,40000]
    con=graph_gens(seed_in=seed_start[ii])
    for jj in range(7500):
    # while True:
        num_props=0
        while num_props<2:
            con.reset()
            num_props, nodes=con.generate_concept()
            # print(nodes)
        # x_current, edge_current = utils.convert2tensor(nodes)
        
        joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes)
        for j in step_height:
            for i in slope:
                final_pos, _, _, b0=env_util.build_steps(sim)
                # final_pos, _, b0 = env_util.build_slope(sim,25)
                # final_pos, _, _, b0=env_util.build_gaussian_field(sim)
                # final_pos, _, _, b0=env_util.build_rough_slope(sim)
                success, time_sim, ave_torque, max_torque, ave_power, max_power, pin, ave_speed = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
                sim.removeObject(b0)
                sim_results.append([success,time_sim,ave_torque,max_torque,ave_power,max_power,pin[0],pin[1],pin[2],ave_speed,j,i])
                x_rec.append(copy.copy(x_current))#, edge_current
                edge_rec.append(copy.copy(edge_current))
                count+=1
                if count_save==10:
                    save_results(x_rec,edge_rec,sim_results,21,ii)
                    count_save=0
                else:
                    count_save+=1
        nodes=[]
        sim.closeScene()
        print(ii, count)

# run_multi(0)
processes = []
for i in range(3):
    p = Process(target=run_multi, args=(i,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()