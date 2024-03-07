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

def input_vectors(edges,nodes,results,terrains,num_bodies=4,num_body_reals=3,num_prop_reals=4,num_joint_reals=4,num_prop_ints=4,num_joint_ints=3):
    data=[]
    # del edges[220]
    # del nodes[220]
    # del results[220]
    for i, edge in enumerate(edges):
        body_reals=np.zeros(num_bodies*num_body_reals)
        prop_reals=np.zeros(num_bodies*num_prop_reals)
        joint_reals=np.zeros((num_bodies)*num_joint_reals)
        body_ints=np.zeros(num_bodies-1)
        prop_ints=np.zeros(num_bodies*num_prop_ints)
        joint_ints=np.zeros(num_bodies*num_joint_ints)
        edge2=[]
        body_id=0
        moveon=False
        count=0
        while moveon==False:
            if nodes[i][body_id+1][0]<5:
                # there is a wheel
                edge2.append([body_id,body_id+1])
                edge2.append([body_id,body_id+2])
                if body_id>0:
                    edge2.append([body_id-1,body_id])                
                body_id+=3
            else:
                edge2.append([body_id,body_id+1])
                if body_id>0:
                    edge2.append([body_id-1,body_id])
                body_id+=2
            if body_id+1>len(nodes[i]):
                moveon=True
        
        if np.array_equal(np.array(edge2,dtype=float),edge)==False:
            edge=np.array(edge2)
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
        props_on_body=np.sum(prop_ints.reshape((4,-1))[:,1:])
        reals=np.hstack((np.hstack((body_reals,prop_reals)),joint_reals[:(num_bodies-1)*num_joint_reals]))
        body_ints[body_count-2]=1
        ints=np.hstack((np.hstack((body_ints,prop_ints)),joint_ints[:(num_bodies-1)*num_joint_ints]))
        x=torch.tensor(np.hstack((np.hstack((reals,ints)),terrains[i])),dtype=torch.float)
        # x=torch.tensor(np.hstack((reals,ints)),dtype=torch.float)
        out_results=np.zeros((3))
        out_results[0]=props_on_body*results[i][2]/250.#250,400,275#np.array()
        out_results[1]=props_on_body*results[i][4]/275#np.array()
        out_results[-1]=-results[i][6]/(results[i][1]+0.1)*100/29      
        # out_results[:4]=np.array(results[i][2:6])
        # out_results[-1]=-results[i][6]/(results[i][1]+0.1)*100
        y=torch.tensor(out_results,dtype=torch.float)
        # y=torch.tensor(results[i],dtype=torch.float)
        # if y[4]>0. or y[6]<0.:
        #     pass
        # else:
        if abs(y[-1])<50/29 and y[-1]>-1./29:
            data.append([x,y])

    # torch.save(data,os.path.join('./','data.pt'))
    return data
            
def create_dataset():
    # num_in=[10,39,43,44]#,6]
    # run_num=[0,1,2,3]
    node_files=['nodes_rough','nodes_steps','nodes_slope','nodes_rough_slope']
    edge_files=['edges_rough','edges_steps','edges_slope','edges_rough_slope']
    result_files=['results_rough','results_steps','results_slope','results_rough_slope']
    terrain_files=['rough_ze.npy','stairs_ze.npy','slope_ze.npy','rough_slope_ze.npy']
    nodes=[]
    edges=[]
    results=[]
    t_rec=[]
    terrains=np.zeros((len(terrain_files),8))
    path='../results/EA_Designs_nopropz/'
    # path='../results/12_10_2023/'
    pre_len=0
    for i in range(len(node_files)):
        terrains[i,:]=np.load('terrain_npy/'+terrain_files[i])
        if len(nodes)==0:
            nodes=read_inputs(path+node_files[i]+'.txt',[])
            edges=read_inputs(path+edge_files[i]+'.txt',[])
            results=read_inputs(path+result_files[i]+'.txt',[])
            for j in range(len(nodes)):
                t_rec.append(terrains[i,:])
            
        else:
            # if i==13 and j==1:
            #     pass
            # else:
            nodes=read_inputs(path+node_files[i]+'.txt',nodes)
            edges=read_inputs(path+edge_files[i]+'.txt',edges)
            results=read_inputs(path+result_files[i]+'.txt',results)
            for j in range(len(nodes)-prev_len):
                t_rec.append(terrains[i,:])
        prev_len=len(nodes)

    return input_vectors(edges,nodes,results,t_rec)

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
        
