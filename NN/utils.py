import torch
import numpy as np
import os

def read_inputs(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    output=[]
    current_output=[]#np.zeros((output_size,attributes))
    count=0
    for i in lines:
        if len(i.replace("\n",""))==0:
            if np.linalg.norm(current_output)>0:
                output.append(np.array(current_output))
            current_output=[]#np.zeros((output_size,attributes))
            count=0
        else:
            val=i.replace("\n", "").replace("[","").replace("]","").split(",")
            current_output.append([float(j) for j in val])
            count+=1
        
    return output

def input_vectors(edges,nodes,num_bodies=4,num_body_reals=3,num_prop_reals=4,num_joint_reals=4,num_prop_ints=4,num_joint_ints=3):
    data=[]
    for i, edge in enumerate(edges):
        body_reals=np.zeros(num_bodies*num_body_reals)
        prop_reals=np.zeros(num_bodies*num_prop_reals)
        joint_reals=np.zeros((num_bodies)*num_joint_reals)
        body_ints=np.zeros(num_bodies-1)
        prop_ints=np.zeros(num_bodies*num_prop_ints)
        joint_ints=np.zeros(num_bodies*num_joint_ints)
        body_id=0
        body_count=0
        moveon=False
        while moveon==False:
            body_reals[body_count*num_body_reals:(body_count+1)*num_body_reals]=nodes[i][body_id,:3]
            if len(np.where(edge[:,0]==body_id)[0])==2:
                ## Set reals ##
                prop_reals[body_count*num_prop_reals:(body_count+1)*num_prop_reals]=nodes[i][body_id+1,1:]
                joint_reals[body_count*num_joint_reals:body_count*num_joint_reals+2]=nodes[i][body_id+2,:2]
                joint_reals[body_count*num_joint_reals+2:(body_count+1)*num_joint_reals]=nodes[i][body_id+2,3:]

                ## Set ints ##
                prop_ints[body_count*num_prop_ints+1+int(nodes[i][body_id+1,0])]=1
                joint_ints[body_count*num_joint_ints+int(nodes[i][body_id+2,2])]=1
                body_id+=3
                body_count+=1
                if len(np.where(edge[:,0]==body_id)[0])==0:
                    moveon=True
            else:
                ## Set reals ##
                joint_reals[body_count*num_joint_reals:body_count*num_joint_reals+2]=nodes[i][body_id+1,:2]
                joint_reals[body_count*num_joint_reals+2:(body_count+1)*num_joint_reals]=nodes[i][body_id+1,3:]

                ## Set ints ##
                joint_ints[body_count*num_joint_ints+int(nodes[i][body_id+1,2])]=1
                prop_ints[body_count*num_prop_ints]=1
                body_id+=2
                body_count+=1
                if len(np.where(edge[:,0]==body_id)[0])==0:
                    moveon=True
        reals=np.hstack((np.hstack((body_reals,prop_reals)),joint_reals[(num_bodies-1)*num_joint_reals:]))
        body_ints[body_count-2]=1
        ints=np.hstack((np.hstack((body_ints,prop_ints)),joint_ints))
        data.append([torch.tensor(np.hstack((reals,ints)),dtype=torch.float)])
    torch.save(data,os.path.join('./','data.pt'))
    return data
            

def create_dataset():
    nodes=read_inputs('nodes.txt')
    edges=read_inputs('edges.txt')
    return input_vectors(edges,nodes)

# create_dataset()