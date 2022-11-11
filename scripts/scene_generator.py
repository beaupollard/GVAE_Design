import time
from zmqRemoteApi import RemoteAPIClient
import utils
import numpy as np
import math
from graph_generator import graph_gens

## Get ID of open CoppeliaSim scene ##
client = RemoteAPIClient()
sim = client.getObject('sim')
motors=[]

con=graph_gens()
nodes=con.generate_concept()
utils.build_vehicles(sim,nodes)
# utils.build_planet_wheels(sim,0.25)
# b0=utils.generate_body(sim,[0.3,0.15,0.2])
# rj0, rw0, lj0, lw0 = utils.build_wheels(sim,0.25)
# rt0, rj0, rj1, lt0, lj0, lj1   = utils.generate_tracks(sim,0.15, 0.45)

# track_link=(get_objects('Track_Link',iter=False)[0])
print('hey')