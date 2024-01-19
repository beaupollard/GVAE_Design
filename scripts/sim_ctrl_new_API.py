import time
from scipy.spatial.transform import Rotation as R
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import copy
import numpy as np
import json

def steering(sim,body_id,motor_ids,radius,velo,des_angle=0.):
    # if sim.getObjectPosition(body_id,-1)[0]<-4.15:
    #     des_angle = math.pi/2
    # sim.setObjectOrientation(body_id,-1,[0.,0.,3.14/4])
    eulerAngles=sim.getObjectOrientation(body_id,-1)
    # r = R.from_euler('xyz', eulerAngles, degrees=False)

    for i,motor in enumerate(motor_ids):
        if i % 2 == 0:
            motor_direction=-1
        else:
            motor_direction=1
        rot_velo=(velo/60/radius[i])+motor_direction*(eulerAngles[-1]-des_angle)
        sim.setJointTargetVelocity(motor.item(),rot_velo)

def end_sim(sim,final_pos,body_id):
    pin=sim.getObjectPosition(body_id,-1)
    pin_ori=sim.getObjectOrientation(body_id,-1)
    if pin_ori[1]*180/math.pi>85 or pin_ori[1]*180/math.pi<-50:
        return True, pin
    if pin[0]<final_pos[0] and pin[2]>final_pos[1]:
        return True, pin
    else:
        return False, pin

def torque_rec(sim,motor_ids,torque,power):
    tor=[]
    pow=[]
    for i in motor_ids:
        tor.append(sim.getJointForce(i.item()))
        pow.append((sim.getJointForce(i.item())*60*sim.getJointVelocity(i.item())/(2*math.pi))/9.5488)
    return torque.append(tor), power.append(pow)

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
    power=[]
    vel_rec=[]
    sim.setInt32Parameter(sim.intparam_dynamic_engine,sim.physics_mujoco)
    # client.setStepping(True)
    sim.setStepping(True)
    sim.startSimulation()

    end_sim_var=False
    count=0

    ## Run simulation ##
    while end_sim_var==False:
        sim.step()
        steering(sim,body_id,motor_ids,radius,velo)
        torque_rec(sim,motor_ids,torque,power)
        success, pin = end_sim(sim,final_pos,body_id)
        vel,omega=sim.getObjectVelocity(body_id)
        mat=np.array(sim.getObjectMatrix(body_id,-1)).reshape((3,4))
        body_vel=mat[:3,:3]@vel   
        vel_rec.append(abs((mat[:3,:3]@vel)[0]))
        if success==True or sim.getSimulationTime()>30.:
            time=sim.getSimulationTime()
            end_sim_var=True
        count+=1 
    if sim.getSimulationTime()<20.:
        ave_vel=0.0
    else:
        ave_vel=np.array(vel_rec).mean()
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
    sum_power=[sum(abs(np.array(power)[i,:])) for i in range(len(power))]

    return success, time, sum(sum_torque)/len(sum_torque), max(sum_torque), sum(sum_power)/len(sum_power), max(sum_power), pin, ave_vel