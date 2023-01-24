import time
from zmqRemoteApi import RemoteAPIClient
import numpy as np
import math
import copy
import numpy as np
import json

def steering(sim,body_id,motor_ids,radius,velo):
    
    # sim.setObjectOrientation(body_id,-1,[0.,0.,3.14/4])
    eulerAngles=sim.getObjectOrientation(body_id,-1)

    for i,motor in enumerate(motor_ids):
        if i % 2 == 0:
            motor_direction=-1
        else:
            motor_direction=1
        rot_velo=(velo/60/radius[i])+motor_direction*eulerAngles[-1]
        sim.setJointTargetVelocity(motor.item(),rot_velo)

def end_sim(sim,final_pos,body_id):
    pin=sim.getObjectPosition(body_id,-1)
    pin_ori=sim.getObjectOrientation(body_id,-1)
    if pin_ori[1]*180/math.pi>85 or pin_ori[1]*180/math.pi<-10:
        return True, pin
    if pin[0]<final_pos[0] and pin[2]>final_pos[1]:
        return True, pin
    else:
        return False, pin

def torque_rec(sim,motor_ids,torque):
    tor=[]
    for i in motor_ids:
        tor.append(sim.getJointForce(i.item()))
    return torque.append(tor)

def set_radius(nodes):
    radius=[]
    prev_track=False
    for i in nodes:
        if i["name"]=='prop':
            if prev_track==True:
                prev_track=False
            else:
                radius.append(i['radius'])
                radius.append(i['radius'])
                if i['type']=='track':
                    prev_track=True  
    return radius 

def main_run(motor_ids,body_id,nodes,final_pos,client,sim):
    
    radius=set_radius(nodes)
    velo=15. 
    torque=[]

    # client.setStepping(True)
    sim.startSimulation()

    end_sim_var=False
    count=0

    ## Run simulation ##
    while end_sim_var==False:
        # client.step()
        steering(sim,body_id,motor_ids,radius,velo)
        torque_rec(sim,motor_ids,torque)
        success, pin = end_sim(sim,final_pos,body_id)
        if success==True or sim.getSimulationTime()>30.:
            time=sim.getSimulationTime()
            end_sim_var=True
        count+=1        
    sim.stopSimulation()
    #     sim.simxSynchronousTrigger(sim_scene.clientID)
    #     sim_scene.steering()
    #     sim_scene.torque_rec()
    #     success=sim_scene.end_sim(final_pos,count*0.05)
    #     if success==True or count*0.05>30.:
    #         end_sim_var=True
    #     count+=1
    # err0=sim.simxStopSimulation(sim_scene.clientID,sim.simx_opmode_oneshot)
    while sim.getSimulationState()!=sim.simulation_stopped:
        pass
    #     # print("Simulation not ending")
    # err1=sim.simxFinish(sim_scene.clientID) # Connect to CoppeliaSim
    # # print(err0,err1)
    sum_torque=[sum(abs(np.array(torque)[i,:])) for i in range(len(torque))]
    return success, time, sum(sum_torque)/len(sum_torque), max(sum_torque), pin