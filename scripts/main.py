import sim
import numpy as np
from utils import get_gauss_rand
import math

def get_objects(name,iter=False):
    ids=[]
    err=0
    i=0
    if iter==True:
        while err==0:
            handle_name=name+str(i)
            err,id = sim.simxGetObjectHandle(clientID,handle_name,sim.simx_opmode_oneshot_wait)
            if err ==0:
                ids.append(id)
                i=i+1

    else:
        handle_name=name
        err,id = sim.simxGetObjectHandle(clientID,handle_name,sim.simx_opmode_oneshot_wait)
        if err ==0:
            ids.append(id)
            i=i+1
    
    if ids==[]:
        print("No objects were found with input name")
    return ids

def set_scene(ids,mean=0,std=0.1,max=1,min=0):
    for i in ids:
        xpos=get_gauss_rand(2.0,0.65,1,3.0)
        ypos=get_gauss_rand(0.,0.4,-0.8,0.8)
        zpos=(2/4)*xpos
        sim.simxSetObjectPosition(clientID,i,-1,[xpos,ypos,zpos],sim.simx_opmode_oneshot)

def set_velocity(ids,velo):
    for i in ids:
        sim.simxSetJointTargetVelocity(clientID,i,velo,sim.simx_opmode_streaming)

def end_sim():
    err,pin=sim.simxGetObjectPosition(clientID,body_id,-1,sim.simx_opmode_buffer)
    pos.append(pin)
    tor=[]
    for i in motor_ids:
        tor.append(sim.simxGetJointForce(clientID,i,sim.simx_opmode_buffer)[-1])
    torque.append(tor)

    if pin[0]>=end_pos or len(pos)>20/0.05:
        np.save('torque.npy',np.array(torque))
        np.save('pos.npy',np.array(pos))        
        return True
    else:
        return False

def steering():
    err,quat=sim.simxGetObjectOrientation(clientID,body_id,-1,sim.simx_opmode_buffer)
    err=quat[-1]*math.pi
    for i,motor in enumerate(motor_ids):
        velo=init_velo+motor_direction[i]*err
        sim.simxSetJointTargetVelocity(clientID,motor,velo,sim.simx_opmode_streaming)

def init_vrep_calls():
    sim.simxGetObjectPosition(clientID,body_id,-1,sim.simx_opmode_streaming)
    sim.simxGetObjectOrientation(clientID,body_id,-1,sim.simx_opmode_streaming)
    for i in motor_ids:
        sim.simxGetJointForce(clientID,i,sim.simx_opmode_streaming)    

def set_max_torque(max_torque=250):
    for i in motor_ids:
        sim.simxSetJointMaxForce(clientID,i,max_torque,sim.simx_opmode_oneshot)

def set_body_link_parms(mode='fixed',max_torque=1000,kp=100,kd=100):
    if mode=='fixed':
        sim.simxSetObjectIntParameter(clientID,bodylink_id,2030,sim.simx_opmode_oneshot)
        sim.simxSetObjectIntParameter(clientID,bodylink_id,2030,0,sim.simx_opmode_oneshot)

## Get ID of open CoppeliaSim scene ##
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
end_pos=4.1
pos=[]
torque=[]

## Get the scene object ids ##
ids=get_objects('obj',iter=True)

## Get the motor ids. Note something is wrong with the python API so 2 motor ids are hard coded. If model is updated you must make sure the motor ids haven't changed. ##
motor_names=['Drive_Front_Right','Drive_Front_Left','Drive_Back_Right','Drive_Back_Left']
motor_direction=np.array([1,1,-1,-1])
motor_ids=[]
for i in motor_names[:2]:
    motor_ids.append(get_objects(i,iter=False)[0])
motor_ids.append(92)
motor_ids.append(95)

## Get the id of the front vehicle frame ##
body_id=get_objects('Frame_1',iter=False)[0]

## Get the id of the joint linking the bodies ##
bodylink_id=get_objects('Body_Link',iter=False)[0]

## Perform initial call ##
init_vrep_calls()

## Initialize 0 joint velocity ##
init_velo=-0*math.pi
set_velocity(motor_ids,init_velo)

## Set simulation to run synchronously ##
sim.simxSynchronous(clientID,True)

## Randomly place objects in scene
set_scene(ids)

## Start a simulation ##
sim.simxStartSimulation(clientID,sim.simx_opmode_oneshot)

## Let blocks fall onto stairs ##
for i in range(10):
    sim.simxSynchronousTrigger(clientID)

## Set track sprocket velocity ##
init_velo=-2*math.pi
set_velocity(motor_ids,init_velo)
set_max_torque(max_torque=100)

end_sim_var=False
## Run simulation ##
while end_sim_var==False:
    sim.simxSynchronousTrigger(clientID)
    steering()
    end_sim_var=end_sim()

## Stop a simulation ##
sim.simxStopSimulation(clientID,sim.simx_opmode_oneshot)

# ## Get an objects handle ##
# err,m1=sim.simxGetObjectHandle(clientID,'Drive_Front_Left',sim.simx_opmode_oneshot_wait)

# ## Set motors joint velocity [rad/s] ##
# sim.simxSetJointTargetVelocity(clientID,m1,-1,sim.simx_opmode_streaming)

# ## Get torque being applied (first call)##
# sim.simxGetJointForce(clientID,m1,sim.simx_opmode_streaming)

# ## Get torque being applied (subsequent calls)##
# sim.simxGetJointForce(clientID,m1,sim.simx_opmode_buffer)