import time
from zmqRemoteApi import RemoteAPIClient
import utils
import numpy as np
import math
from graph_generator import graph_gens
from sim_ctrl import main_run
import copy
import numpy as np

## Get ID of open CoppeliaSim scene ##
client = RemoteAPIClient()
sim = client.getObject('sim')
motors=[]
sim.closeScene()
sim.setBoolParam(sim.boolparam_display_enabled,False)
x_rec=[]
edge_rec=[]
sim_results=[]
client_id=0
# for i in range(100):
while True:
    num_props=0
    while num_props<2:
        con=graph_gens()
        num_props, nodes=con.generate_concept()
    x_current, edge_current = utils.convert2tensor(nodes)
    
    joints, body_id = utils.build_vehicles(sim,nodes)
    final_pos=utils.build_steps(sim)
    success, time, client_id = main_run(np.array(joints).flatten(),body_id,nodes,final_pos,client_id,sim)
    sim.closeScene()
    sim_results.append([success,time])
    x_rec.append(copy.copy(x_current)), edge_current
    edge_rec.append(copy.copy(edge_current))


with open('nodes2.txt', 'w') as convert_file:
    for i in x_rec:
        for j in i:

            convert_file.write(json.dumps(j))
            convert_file.write("\n")
        convert_file.write("\n")
        convert_file.write("\n")

with open('edges2.txt', 'w') as convert_file:
    for i in edge_rec:
        for j in i:

            convert_file.write(json.dumps(j))
            convert_file.write("\n")
        convert_file.write("\n")
        convert_file.write("\n")

with open('results2.txt', 'w') as convert_file:
    for i in sim_results:
        convert_file.write(json.dumps(i))
        convert_file.write("\n")

# print('hey')
# utils.build_planet_wheels(sim,0.25)
# b0=utils.generate_body(sim,[0.3,0.15,0.2])
# rj0, rw0, lj0, lw0 = utils.build_wheels(sim,0.25)
# rt0, rj0, rj1, lt0, lj0, lj1   = utils.generate_tracks(sim,0.15, 0.45)

# track_link=(get_objects('Track_Link',iter=False)[0])
