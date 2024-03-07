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
import pickle

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
    if terrain == 1:
        final_pos, _, _, b0=env_util.build_steps(sim)
    elif terrain == 2:
        final_pos, _, b0 = env_util.build_slope(sim,25)
    elif terrain == 0:
        final_pos, _, _, b0=env_util.build_gaussian_field(sim)
    elif terrain == 3:
        final_pos, _, _, b0=env_util.build_rough_slope(sim)
    else:
        final_pos, _, _, b0=env_util.build_gaussian_field_obs(sim)
    return final_pos, b0

terrain = 0
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

n_samples=25
pool_nodes = []
f_obj=np.array([0.8,0.1,0.1])
## Generate the initial population ##
t_list=[3]#[0,1,2,3]
seeds=[25]#[10,20,30]

for seed_i in seeds:
    for terrain in t_list:
        sim_results=[]
        pool_nodes = []
        Jmax=-100.        
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
            num_props=0
            for j in nodes_gen:
                if j['name']=="prop" and j['type']!='none':
                    num_props+=1    
            joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes_gen)
            final_pos, b0 = select_terrain()
            success, time_sim, ave_torque, max_torque, ave_power, max_power, pin, ave_vel = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
            torque=num_props*ave_torque/250.
            power=num_props*ave_power/275.
            vel=-pin[0]/(time_sim+0.1)*100/29.#ave_vel*100/29

            fit_func[i]=(vel*f_obj[0]-torque*f_obj[1]-power*f_obj[2])
            if fit_func[i]>Jmax:
                Jmax=copy.copy(fit_func[i])
                nodes_max = copy.copy(nodes_gen)

            sim_results.append([success,time_sim,ave_torque,max_torque,ave_power,max_power,pin[0],pin[1],pin[2],ave_vel,i])
            x_rec.append(copy.copy(x_current))#, edge_current
            edge_rec.append(copy.copy(edge_current))   
            sim.removeObject(b0)
            sim.closeScene()
        # for jj in range(7500):
        mean_fit=[]
        counter=0
        while len(sim_results)<100:
            fit_rec=[]

            ## Generate crossover designs ##
            new_des=utils.crossover(copy.copy(pool_nodes))
            ## Mutate the pool_nodes ##
            mut_des=utils.mutation(copy.deepcopy(pool_nodes),fit_func)

            for i, nodes_gen in enumerate(new_des+mut_des):
                num_props=0
                for j in nodes:
                    if j['name']=="prop" and j['type']!='none':
                        num_props+=1            
                joints, body_id, x_current, edge_current, nodes = utils.build_vehicles(sim,nodes_gen)
                final_pos, b0 = select_terrain()
                success, time_sim, ave_torque, max_torque, ave_power, max_power, pin, ave_vel  = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client,sim)
                torque=num_props*ave_torque/250.
                power=num_props*ave_power/275.
                vel=-pin[0]/(time_sim+0.1)*100/29.#ave_vel*100/29
                fit_rec.append(vel*f_obj[0]-torque*f_obj[1]-power*f_obj[2])
                # fit_func[i]=(vel*f_obj[0]-torque*f_obj[1]-power*f_obj[2])
                if (vel*f_obj[0]-torque*f_obj[1]-power*f_obj[2])>Jmax:
                    Jmax=copy.copy((vel*f_obj[0]-torque*f_obj[1]-power*f_obj[2]))
                    nodes_max = copy.copy(nodes_gen)                
                sim.removeObject(b0)
                sim.closeScene()
                sim_results.append([success,time_sim,ave_torque,max_torque,ave_power,max_power,pin[0],pin[1],pin[2],ave_vel,i])
                x_rec.append(copy.copy(x_current))#, edge_current
                edge_rec.append(copy.copy(edge_current))        
                counter+=1
                print(counter,Jmax)
                # yout=out_results@obj
        sim_results=np.array(sim_results)
        out_results=np.zeros((len(sim_results),3))
        for i in range(len(out_results)):
            out_results[i,0]=sim_results[i,2]/250.
            out_results[i,1]=sim_results[i,4]/275.
            out_results[i,2]=-sim_results[i,6]/(sim_results[i,1]+0.1)*100/29.
        
        yout=out_results@np.array([-f_obj[1],-f_obj[2],f_obj[0]])
        np.argmax(yout)
        data = {
            'terrain' : terrain,
            'obj' : np.array([-f_obj[1],-f_obj[2],f_obj[0]]),
            
            'ysamples' : yout,
            'nodes' : nodes_max,
            'Jmax' : Jmax
        }

        ## Print Dictionary to file
        with open('0219_data_EA_'+str(terrain)+'_'+str(seed_i)+'.pkl', 'wb') as fp:
            pickle.dump(data, fp)
            print('dictionary saved successfully to file')          


    # save_results(x_rec,edge_rec,sim_results,21,'wall_passage_seed_'+str(seed)+'terrain_'+str(terrain))
    # ## Select best samples ##
    # total_fit=np.concatenate((fit_func,np.array(fit_rec)))
    # fit_index=np.argsort(total_fit)[::-1]
    # total_nodes=copy.copy(pool_nodes+new_des+mut_des)
    # for i in range(len(pool_nodes)):
    #     pool_nodes[i]=copy.copy(total_nodes[fit_index[i]])
    #     fit_func[i]=copy.copy(total_fit[fit_index[i]])
    # mean_fit.append(np.mean(fit_func))
    # if len(mean_fit)>2:
    #     if mean_fit[-2]==mean_fit[-1]:
    #         break
    # print(mean_fit[-1])