import numpy as np
import copy
import random
import math 

def set_length_location(min_x=-100,body_num=4,BLprev=0,bodyw=0,bodyh=0,zloc=0,yloc=0):
    err=False
    flag=False
    count=0
    while err==False:
        
        body_length=36./body_num/39.37#get_gauss_rand(30./body_num,12,24/body_num,36./body_num)/39.37
        prop_radius=get_gauss_rand(8.,4.,4.,10.)/39.37

        prop_location=[get_gauss_rand(0,5.,-body_length/2,body_length/2)/39.37,bodyw+0.15,-bodyh/2+0.05+get_gauss_rand(0.,5,-5,1)/39.37]
        joint_location=[body_length/2+1/39.37,0,get_gauss_rand(0.,5,-5,5)/39.37]
        R0=BLprev/2+2/39.37+body_length/2
        overlap=R0+prop_location[0]-prop_radius
        if overlap>1.3*min_x and body_length/2>prop_radius:
            err=True
        if count>1000:
            flag=True
            err=True
        count+=1
    min_x=prop_location[0]+prop_radius
    BLprev=body_length
    
    return body_length, prop_radius, prop_location, joint_location, BLprev, min_x, flag

def set_tracks(body_num=4,bodyw=0,bodyh=0,zloc=0,yloc=0):
    err=False
    count=0
    flag=False
    while err==False:    
        # body_length=get_gauss_rand(30./body_num,12,24/body_num,36./body_num)/39.37
        body_length=36./body_num/39.37#get_gauss_rand(30./body_num,12,24/body_num,36./body_num)/39.37
        prop_location=[get_gauss_rand(0,5.,-body_length,body_length)/39.37,bodyw+0.15,-bodyh/2+0.05]
        prop_location2=[get_gauss_rand(0,5.,-body_length,body_length)/39.37,bodyw+0.15,-bodyh/2+0.05]
        joint_location=[body_length/2+1/39.37,0,get_gauss_rand(0.,5,-5,5)/39.37]
        prop_radius=4.5/39.37#get_gauss_rand(8.,4.,3.,10.)/39.37
        body_length2=get_gauss_rand(12.,4.,6.,14.)/39.37
        wheel_base=joint_location[0]+body_length2/2+prop_location2[0]-prop_location[0]
        if wheel_base>2.35*prop_radius:
            err=True
        if count>1000:
            flag=True
            err=True
        count+=1
    return body_length, body_length2, prop_location, prop_location2, joint_location, prop_radius, flag

def get_gauss_rand(in_mean,in_std=0,l_lim=-1000000,u_lim=1000000):
    outp = l_lim-1
    while outp<l_lim or outp>u_lim:
        outp = np.random.normal(in_mean,in_std)
    return outp

class graph_gens():
    def __init__(self,seed_in=0):
        np.random.seed(seed=seed_in)
        random.seed(seed_in)
        self.reset()

    def reset(self):        
        self.nodes=[]
        self.node_attributes=[]
        self.edge_index=[]
        self.body_nodes={"name":"body","location":[],"length": 0,"width":20/39.37,"height":12/39.37,"clearance":0,"childern": [],"parents": [],"index":0}
        self.joint_nodes={"name":"joint","location": [0, 0, 0],"orientation":[0,0,0],"active":[],"childern": [],"parents": [],"index":1}
        self.prop_nodes={"name":"prop","location": [0, 0, 0],"radius":0,"childern": [],"parents": [],"type":'none'}
        self.num_propulsors=0


    def generate_concept(self):
        ## Determine number of bodies ##
        body_num=random.randint(2, 4)

        prev_tracked=False      # set that the previous mechanism wasn't a track
        current_node=0          # current graph node
        min_x=-10               # where there will be overlap in wheels
        BLprev=0
        prop_types=['none','wheel','planet wheel','track']

        for i in range(body_num):
            ## Initialize the node types ##
            body=copy.copy(self.body_nodes)
            joint=copy.copy(self.joint_nodes)
            propulsors=copy.copy(self.prop_nodes )           

            ## Set the body node parent (previous joint) ##
            if i==0:
                body['parents']=[]
            else:
                body['parents']=current_node-1

            ## Determine if the next joint is active ##
            if random.randint(0, 1)==1:
                joint['active']=[get_gauss_rand(200,50,100,400),get_gauss_rand(30,10,15,40)]
            else:
                joint['active']=[1000,5]
            joint['parents']=current_node

            ## Joint orientation ##
            joint['orientation']=[0.,0.,0.]
            joint['orientation'][random.randint(0, 2)]=math.sin(math.pi/4)

            ## Determine what type of propulsion ##
            if prev_tracked==True:  
                propulsors["type"]='track'
                prev_tracked=False
                body["length"]=copy.copy(body_length2)
                propulsors["radius"]=copy.copy(prop_radius)
                joint["location"]=[body["length"]/2+1/39.37,0,get_gauss_rand(0.,5,-5,5)/39.37]
                propulsors["location"]=copy.copy(prop_location2)
                min_x=copy.copy(prop_location2[0])+copy.copy(prop_radius)
                BLprev=copy.copy(body_length2)
            else:
                ## Determine what type of mechanism is next [0=none, 1=wheel, 2=planet wheel, 3=track] ##
                if i==body_num-1:
                    propulsors["type"]=prop_types[random.randint(0, 2)]
                elif i==0:
                    propulsors["type"]=prop_types[random.randint(1, 3)]
                else:
                    propulsors["type"]=prop_types[random.randint(0, 3)]
            
            
                if propulsors["type"]!='track':
                    body["length"], propulsors["radius"], propulsors["location"], joint["location"], BLprev, min_x, flag = set_length_location(min_x=min_x,body_num=body_num,BLprev=BLprev,bodyw=body["width"],bodyh=body["height"],zloc=0,yloc=0)

                else:
                    body["length"], body_length2, propulsors["location"], prop_location2, joint["location"], prop_radius, flag = set_tracks(body_num=body_num,bodyw=body["width"],bodyh=body["height"],zloc=0,yloc=0)
                    propulsors["radius"]=copy.copy(prop_radius)
                    prev_tracked=True
                    joint['orientation']=[0.,0.,0.]
                    joint['orientation'][1]=math.sin(math.pi/4)

            if propulsors["type"]==prop_types[0]:
                body["childern"]=[current_node+1]
                self.nodes.append(copy.copy(body))
                self.nodes.append(copy.copy(joint))
                current_node+=2
            else:
                if i==0:
                    propulsors["location"][0]=-body['length']/2+propulsors['radius']*0.9
                body["childern"]=[current_node+1,current_node+2]
                propulsors["parents"]=current_node
                self.nodes.append(copy.copy(body))
                self.nodes.append(copy.copy(propulsors))      
                self.nodes.append(copy.copy(joint))  
                self.num_propulsors+=1              
                current_node+=3
            if flag==True:
                self.num_propulsors=-10
        return self.num_propulsors, self.nodes
            
con=graph_gens()
con.generate_concept()