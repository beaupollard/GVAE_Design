import sim
import numpy as np
from utils import get_gauss_rand
import math


class run_sim():
    def __init__(self,motor_ids,body_id,nodes,jj):
        self.motor_ids=motor_ids
        self.body_id=body_id
        # if jj==0:
        self.clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
        # else:
        #     self.clientID=jj
        prev_track=False
        self.radius=[]
        self.torque=[]
        self.velo=15.
        for i in nodes:
            if i["name"]=='prop':
                if prev_track==True:
                    prev_track=False
                else:
                    self.radius.append(i['radius'])
                    self.radius.append(i['radius'])
                    if i['type']=='track':
                        prev_track=True

    def init_vrep_calls(self):
        sim.simxGetObjectPosition(self.clientID,self.body_id,-1,sim.simx_opmode_streaming)
        sim.simxGetObjectOrientation(self.clientID,self.body_id,-1,sim.simx_opmode_streaming)
        for i in self.motor_ids:
            sim.simxGetJointForce(self.clientID,i,sim.simx_opmode_streaming)

    def set_max_torque(self,max_torque=150):
        for i in self.motor_ids:
            sim.simxSetJointMaxForce(self.clientID,i,max_torque,sim.simx_opmode_oneshot)

    def set_velocity(self,):

        for i, joints in enumerate(self.motor_ids):
            rot_velo=(self.velo/60/self.radius[i])
            sim.simxSetJointTargetVelocity(self.clientID,joints,rot_velo,sim.simx_opmode_streaming)

    def steering(self):
        err,quat=sim.simxGetObjectOrientation(self.clientID,self.body_id,-1,sim.simx_opmode_buffer)
        err=quat[-1]*math.pi
        for i,motor in enumerate(self.motor_ids):
            if i % 2 == 0:
                motor_direction=-1
            else:
                motor_direction=1
            rot_velo=(self.velo/60/self.radius[i])+motor_direction*err
            sim.simxSetJointTargetVelocity(self.clientID,motor,rot_velo,sim.simx_opmode_streaming)

    def end_sim(self,final_pos,time):
        err,pin=sim.simxGetObjectPosition(self.clientID,self.body_id,-1,sim.simx_opmode_buffer)
        if pin[0]<final_pos[0] and pin[2]>final_pos[1]:
            success=True
        else:
            success=False
        
        return success
    def torque_rec(self):
        tor=[]
        for i in self.motor_ids:
            tor.append(sim.simxGetJointForce(self.clientID,i,sim.simx_opmode_buffer)[-1])
        self.torque.append(tor)

def main_run(motor_ids,body_id,nodes,final_pos,client_id,simapi):

    sim_scene=run_sim(motor_ids,body_id,nodes,client_id)
    sim_scene.init_vrep_calls()
    sim_scene.set_max_torque(max_torque=150)
    sim_scene.set_velocity()

    ## Set simulation to run synchronously ##
    (sim.simxSynchronous(sim_scene.clientID,True))
    # sim.simxSetBoolParam(sim_scene.clientID,simapi.boolparam_display_enabled,False,sim.simx_opmode_oneshot)
    ## Start a simulation ##
    sim.simxStartSimulation(sim_scene.clientID,sim.simx_opmode_oneshot)

    end_sim_var=False
    count=0
    ## Run simulation ##
    while end_sim_var==False:
        sim.simxSynchronousTrigger(sim_scene.clientID)
        sim_scene.steering()
        sim_scene.torque_rec()
        success=sim_scene.end_sim(final_pos,count*0.05)
        if success==True or count*0.05>30.:
            end_sim_var=True
        count+=1
    err0=sim.simxStopSimulation(sim_scene.clientID,sim.simx_opmode_oneshot)
    while simapi.getSimulationState()!=simapi.simulation_stopped:
        pass
        # print("Simulation not ending")
    err1=sim.simxFinish(sim_scene.clientID) # Connect to CoppeliaSim
    # print(err0,err1)
    sum_torque=[sum(abs(np.array(sim_scene.torque)[i,:])) for i in range(len(sim_scene.torque))]
    return success, count*0.05, sim_scene.clientID, sum(sum_torque)/len(sum_torque), max(sum_torque)

# main_run(motor_ids=[0],body_id=[1],nodes=0,final_pos=[-10,0,2],client_id)