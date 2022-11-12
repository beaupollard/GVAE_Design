import sim
import numpy as np
from utils import get_gauss_rand
import math


class run_sim():
    def __init__(self,motor_ids,body_id,nodes):
        self.motor_ids=motor_ids
        self.body_id=body_id
        self.clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
        prev_track=False
        self.radius=[]
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

def main_run(motor_ids,body_id,nodes):
    sim_scene=run_sim(motor_ids,body_id,nodes)
    sim_scene.init_vrep_calls()
    sim_scene.set_max_torque(max_torque=150)
    sim_scene.set_velocity()

    ## Set simulation to run synchronously ##
    sim.simxSynchronous(sim_scene.clientID,True)

    ## Start a simulation ##
    sim.simxStartSimulation(sim_scene.clientID,sim.simx_opmode_oneshot)

    end_sim_var=False
    ## Run simulation ##
    while end_sim_var==False:
        sim.simxSynchronousTrigger(sim_scene.clientID)
        sim_scene.steering()
        # end_sim_var=end_sim()

