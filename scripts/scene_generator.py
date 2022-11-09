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

def generate_tracks(radius, wheel_base):
    link_width=0.055
    link_height=0.009/2.
    joint_length=0.034
    linkandjoint_width=2*joint_length                                # joint and track length

    track_link=(get_objects('Track_Link',iter=False)[0])            # parent track id
    track_joint=(get_objects('Track_Joint',iter=False)[0])          # parent track joint id
    track_length=2*wheel_base+2*math.pi*(radius+link_height)        # total track length
    num_tracks=math.floor(track_length/linkandjoint_width)          # number of tracks

    if (num_tracks % 2) != 0:
        num_tracks=num_tracks-1

    joint_length=track_length/(2*num_tracks)#1/num_tracks*(track_length-num_tracks*link_width/2)# calculate new joint lengths

    ## Adjust joint length so that distances are consistent ##
    sim.simxSetObjectPosition(clientID,track_joint,track_link,[0,joint_length,0],sim.simx_opmode_oneshot)
    
    ## Get Dummy Ids ##
    dummy_ids=[get_objects('Dummy0',iter=False)[0],get_objects('Dummy1',iter=False)[0]]
    sim.simxSetObjectPosition(clientID,dummy_ids[0],track_link,[0,-joint_length,0],sim.simx_opmode_oneshot)
    sim.simxSetObjectPosition(clientID,dummy_ids[1],track_link,[0,-joint_length,0],sim.simx_opmode_oneshot)

    ## Copy and paste the required number of links ##
    for i in range(num_tracks-1):
        sim.simxCopyPasteObjects(clientID,[track_link,track_joint],sim.simx_opmode_blocking)
    
    ## Get the new track link ids ##
    track_links=(get_objects('Track_Link',iter=True))

    ## Set the track length parent/child relationship ##
    sim.simxSetObjectParent(clientID,track_links[0],track_joint,True,sim.simx_opmode_oneshot)
    sim.simxSetObjectPosition(clientID,track_links[0],track_joint,[0,joint_length,0],sim.simx_opmode_oneshot)
    for i in range(1,len(track_links)):
        sim.simxSetObjectParent(clientID,track_links[i],track_links[i-1]+1,True,sim.simx_opmode_oneshot)
        sim.simxSetObjectPosition(clientID,track_links[i],track_links[i-1]+1,[0,joint_length,0],sim.simx_opmode_oneshot)
    
    ## Determine how many links are needed to surround the drives ##
    linkandjoint_width=2*joint_length#link_width/2+joint_length
    num_tracks_circ=math.ceil(math.pi*(radius+link_height)/linkandjoint_width)
    theta=math.pi/num_tracks_circ
    for i in range(num_tracks_circ):
        sim.simxSetObjectQuaternion(clientID,track_links[i]+1,track_links[i]+1,[0,0,-math.sin(theta/2),math.cos(theta/2)],sim.simx_opmode_oneshot)
    i+=int((num_tracks-2*num_tracks_circ)/2)+1

    for j in range(num_tracks_circ):
        sim.simxSetObjectQuaternion(clientID,track_links[i+j]+1,track_links[i+j]+1,[0,0,-math.sin(theta/2),math.cos(theta/2)],sim.simx_opmode_oneshot)
    
    sim.simxSetObjectParent(clientID,dummy_ids[-1],track_links[-1]+1,True,sim.simx_opmode_oneshot)

    ## Get the positions and sizes of the drive wheels ##
    drive_wheels=(get_objects('Track_Drive',iter=False))
    print('hey')

# def set_scene(ids,mean=0,std=0.1,max=1,min=0):
#     for i in ids:
#         xpos=get_gauss_rand(2.0,0.65,1,3.0)
#         ypos=get_gauss_rand(0.,0.4,-0.8,0.8)
#         zpos=(2/4)*xpos
#         sim.simxSetObjectPosition(clientID,i,-1,[xpos,ypos,zpos],sim.simx_opmode_oneshot)

# def set_velocity(ids,velo):
#     for i in ids:
#         sim.simxSetJointTargetVelocity(clientID,i,velo,sim.simx_opmode_streaming)

# def end_sim():
#     err,pin=sim.simxGetObjectPosition(clientID,body_id,-1,sim.simx_opmode_buffer)
#     pos.append(pin)
#     tor=[]
#     for i in motor_ids:
#         tor.append(sim.simxGetJointForce(clientID,i,sim.simx_opmode_buffer)[-1])
#     torque.append(tor)

#     if pin[0]>=end_pos or len(pos)>20/0.05:
#         np.save('torque.npy',np.array(torque))
#         np.save('pos.npy',np.array(pos))        
#         return True
#     else:
#         return False

# def steering():
#     err,quat=sim.simxGetObjectOrientation(clientID,body_id,-1,sim.simx_opmode_buffer)
#     err=quat[-1]*math.pi
#     for i,motor in enumerate(motor_ids):
#         velo=init_velo+motor_direction[i]*err
#         sim.simxSetJointTargetVelocity(clientID,motor,velo,sim.simx_opmode_streaming)

# def init_vrep_calls():
#     sim.simxGetObjectPosition(clientID,body_id,-1,sim.simx_opmode_streaming)
#     sim.simxGetObjectOrientation(clientID,body_id,-1,sim.simx_opmode_streaming)
#     for i in motor_ids:
#         sim.simxGetJointForce(clientID,i,sim.simx_opmode_streaming)    

# def set_max_torque(max_torque=250):
#     for i in motor_ids:
#         sim.simxSetJointMaxForce(clientID,i,max_torque,sim.simx_opmode_oneshot)

# def set_body_link_parms(mode='fixed',max_torque=1000,kp=100,kd=100):
#     if mode=='fixed':
#         sim.simxSetObjectIntParameter(clientID,bodylink_id,2030,sim.simx_opmode_oneshot)
#         sim.simxSetObjectIntParameter(clientID,bodylink_id,2030,0,sim.simx_opmode_oneshot)

## Get ID of open CoppeliaSim scene ##
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
generate_tracks(0.15, 0.3)
# track_link=(get_objects('Track_Link',iter=False)[0])
# print('hey')