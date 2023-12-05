import torch
import numpy as np
import os

def read_inputs(filename,output):
    with open(filename) as f:
        lines = f.readlines()
    
    # output=[]
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
            if val[0]=='false':
                val[0]=str(0)
                output.append([float(j) for j in val])
            elif val[0]=='true':
                val[0]=str(1)
                output.append([float(j) for j in val])
            current_output.append([float(j) for j in val])
            count+=1
        
    return output

def input_vectors(edges,nodes,results,num_bodies=4,num_body_reals=3,num_prop_reals=4,num_joint_reals=4,num_prop_ints=4,num_joint_ints=3):
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
        reals=np.hstack((np.hstack((body_reals,prop_reals)),joint_reals[:(num_bodies-1)*num_joint_reals]))
        body_ints[body_count-2]=1
        ints=np.hstack((np.hstack((body_ints,prop_ints)),joint_ints[:(num_bodies-1)*num_joint_ints]))
        x=torch.tensor(np.hstack((reals,ints)),dtype=torch.float)
        y=torch.tensor(results[i],dtype=torch.float)
        if y[4]>0. or y[6]<0.:
            pass
        else:
            data.append([x,y])

    # torch.save(data,os.path.join('./','data.pt'))
    return data
            
def create_dataset():
    num_in=[10,39,43,44]#,6]
    run_num=[0,1,2,3]
    nodes=[]
    edges=[]
    results=[]
    path='../results/Current_Results/'
    for i in num_in:
        for j in run_num:
            if len(nodes)==0:
                nodes=read_inputs(path+'nodes'+str(39)+'_'+str(j)+'.txt',[])
                edges=read_inputs(path+'edges'+str(10)+'_'+str(j)+'.txt',[])
                results=read_inputs(path+'results'+str(i)+'_'+str(j)+'.txt',[])
            else:
                # if i==13 and j==1:
                #     pass
                # else:
                nodes=read_inputs(path+'nodes'+str(39)+'_'+str(j)+'.txt',nodes)
                edges=read_inputs(path+'edges'+str(10)+'_'+str(j)+'.txt',edges)
                results=read_inputs(path+'results'+str(i)+'_'+str(j)+'.txt',results)
            # if len(results)!=len(edges):
            #     print(i)
    return input_vectors(edges,nodes,results)

def create_vehicles(x_reals,x_ints,num_bodies=4,num_body_reals=3,num_prop_reals=4,num_joint_reals=4,num_prop_ints=4,num_joint_ints=3):
    ## Determine the number of bodies ##
    bodies=np.argmax(x_ints[:3])+2
    body_reals=np.reshape(x_reals[:num_bodies*num_body_reals],(num_bodies,num_body_reals))
    prop_reals=np.reshape(x_reals[num_bodies*num_body_reals:num_bodies*(num_body_reals+num_prop_reals)],(num_bodies,num_prop_reals))
    joint_reals=np.reshape(x_reals[num_bodies*(num_body_reals+num_prop_reals):],(num_bodies-1,num_joint_reals))
    prop_ints=np.reshape(x_ints[3:3+num_prop_ints*num_bodies],(num_bodies,num_prop_ints))
    joint_ints=np.reshape(x_ints[3+num_prop_ints*num_bodies:],(num_bodies-1,num_joint_ints))
    nodes=[]
    edges=[]
    body_id=0
    for i in range(bodies):
        propid=np.argmax(prop_ints[i,:])
        jointid=np.argmax(joint_ints[i,:])
        nodes.append([body_reals[i,0],body_reals[i,1],body_reals[i,1],0.,0.])
        if propid!=0:
            nodes.append([propid-1, prop_reals[i,0], prop_reals[i,1], prop_reals[i,2], prop_reals[i,3]])
            edges.append([body_id,body_id+1])
            edges.append([body_id,body_id+2])
            index_increase=3
        else:
            edges.append([body_id,body_id+1])
            index_increase=2
        
        nodes.append([joint_reals[i,0],joint_reals[i,1],jointid,joint_reals[i,2],joint_reals[i,3]])
        body_id+=index_increase
        
