import numpy as np
import math
import copy
from scipy.spatial.transform import Rotation as R


def get_gauss_rand(in_mean,in_std=0,l_lim=-1000000,u_lim=1000000):
    outp = l_lim-1
    while outp<l_lim or outp>u_lim:
        outp = np.random.normal(in_mean,in_std)
    return outp

def generate_body(sim,body_size=[0.3,0.15,0.2]):
    density=50*16.0184634 #lbs/ft^3
    volume=body_size[0]*body_size[1]*(body_size[2]+2/39.37)
    b0=sim.createPrimitiveShape(sim.primitiveshape_cuboid,body_size)
    sim.setObjectInt32Param(b0,sim.shapeintparam_static,0)
    sim.setObjectInt32Param(b0,sim.shapeintparam_respondable,1)
    b1=sim.createPrimitiveShape(sim.primitiveshape_spheroid,[0.05,0.05,0.05])
    sim.setShapeMass(b1,volume*density)
    sim.setObjectParent(b1,b0,True)
    sim.setObjectPosition(b1,b0,[0,0,-0.2])
    return b0

def generate_tracks(sim,radius,wheel_base,current_body,jlocation=[0,0,0]):
    
    link_length=0.055
    link_height=0.009/2.
    joint_length=0.034
    track_links=[]
    track_joints=[]
    tl, tj, dummy_ids = build_links(sim,link_length,link_height*2,joint_length)
    sim.isDynamicallyEnabled(tl)

    track_links.append(tl)
    track_joints.append(tj)

    linkandjoint_width=2*joint_length                               # joint and track length


    track_length=2*wheel_base+2*math.pi*(radius+link_height)        # total track length
    num_tracks=math.floor(track_length/linkandjoint_width)          # number of tracks

    if (num_tracks % 2) != 0:
        num_tracks=num_tracks-1

    joint_length=track_length/(2*num_tracks)#1/num_tracks*(track_length-num_tracks*link_width/2)# calculate new joint lengths

    ## Adjust joint length so that distances are consistent ##
    sim.setObjectPosition(track_joints[0],track_links[0],[0,joint_length,0])
    
    # ## Get Dummy Ids ##
    sim.setObjectPosition(dummy_ids[0],track_links[0],[0,-joint_length,0])
    sim.setObjectPosition(dummy_ids[1],track_links[0],[0,-joint_length,0])

    ## Copy and paste the required number of links ##
    for i in range(num_tracks-1):
        ids=sim.copyPasteObjects([track_links[0],track_joints[0]],0)
        track_links.append(ids[0])
        track_joints.append(ids[1])

    ## Set the track length parent/child relationship ##
    for i in range(1,len(track_links)):
        sim.setObjectParent(track_links[i],track_joints[i-1],True)
        sim.setObjectPosition(track_links[i],track_joints[i-1],[joint_length,0,0])
    
    ## Determine how many links are needed to surround the drives ##
    linkandjoint_width=2*joint_length
    num_tracks_circ=math.floor(math.pi*(radius+link_height)/linkandjoint_width)
    theta=math.pi/num_tracks_circ
    for i in range(num_tracks_circ):
        sim.setObjectQuaternion(track_joints[i],track_joints[i],[0,0,math.sin(theta/2),math.cos(theta/2)])
    i+=int((num_tracks-2*num_tracks_circ)/2)+1

    for j in range(num_tracks_circ):
        sim.setObjectQuaternion(track_joints[i+j],track_joints[i+j],[0,0,math.sin(theta/2),math.cos(theta/2)])
    
    sim.setObjectParent(dummy_ids[-1],track_joints[-1],True)
    Rj0, Rj1, Rw0, Rw1, wheel_base, radius = build_track_wheels(sim,track_links[0],track_links[i],num_tracks_circ,radius)
    
    Ltrack_links=sim.copyPasteObjects(track_links+track_joints+dummy_ids,0)
    Lwheels = sim.copyPasteObjects([Rj0,Rj1,Rw0,Rw1],0)
    sim.setObjectParent(Rj1,current_body,0)

    ## Set front sprocket as parent just to get things into position
    sim.setObjectParent(Rj0,Rj1,0)
    sim.setObjectParent(track_links[0],Rj1,0)
    sim.setObjectParent(Lwheels[0],Lwheels[1],0)
    sim.setObjectParent(Ltrack_links[0],Lwheels[1],0)

    sim.setObjectParent(Rj1,current_body,0)
    sim.setObjectPosition(Rj1,current_body,[jlocation[0],jlocation[1]/2,jlocation[2]])
    sim.setObjectParent(Lwheels[1],current_body,0)
    sim.setObjectPosition(Lwheels[1],current_body,[jlocation[0],-jlocation[1]/2,jlocation[2]])

    sim.setObjectParent(track_links[0],current_body,0)
    sim.setObjectParent(Ltrack_links[0],current_body,0)
    set_joint_mode(sim,[Rj0,Lwheels[0]],j_spring=[],jtype='velocity')

    ## Build linear track support ##
    # support_size=[abs(wheel_base)-2.05*radius,0.015,0.025]
    # s0=sim.createPrimitiveShape(sim.primitiveshape_cuboid,support_size)
    # sim.setObjectInt32Param(s0,sim.shapeintparam_respondable,1)   
    # sim.setObjectParent(s0,current_body,0)
    # midp=(sim.getObjectPosition(Rj0,-1)[0]+sim.getObjectPosition(Rj1,-1)[0])/2
    # sim.setObjectPosition(s0,-1,[midp,sim.getObjectPosition(Rj0,-1)[1]+0.025,sim.getObjectPosition(Rj0,-1)[2]-radius+support_size[2]/1.25])
    # s1 = sim.copyPasteObjects([s0],0)
    # sim.setObjectPosition(s1[0],-1,[(sim.getObjectPosition(Rj0,-1)[0]+sim.getObjectPosition(Rj1,-1)[0])/2,sim.getObjectPosition(Rj0,-1)[1]-0.025,sim.getObjectPosition(Rj0,-1)[2]-radius+support_size[2]/1.25])
    # s2 = sim.copyPasteObjects([s0,s1[0]],0)
    # sim.setObjectPosition(s2[0],-1,[sim.getObjectPosition(s0,-1)[0],-sim.getObjectPosition(s0,-1)[1],sim.getObjectPosition(s0,-1)[2]])
    # sim.setObjectPosition(s2[1],-1,[sim.getObjectPosition(s1[0],-1)[0],-sim.getObjectPosition(s1[0],-1)[1],sim.getObjectPosition(s1[0],-1)[2]])
    # sim.setObjectParent(s1[0],current_body,0)
    # sim.setObjectParent(s2[0],current_body,0)
    # sim.setObjectParent(s2[1],current_body,0)

    ## Build rotational track support ##
    length_s=(abs(wheel_base)-2.05*radius)/2
    support_size=[radius/1.5,radius/1.5,0.015]
    s0=sim.createPrimitiveShape(sim.primitiveshape_cylinder,support_size)
    sim.setObjectQuaternion(s0,s0,[math.sin(math.pi/4),0.,0.,math.cos(math.pi/4)])
    sim.setObjectInt32Param(s0,sim.shapeintparam_respondable,1)   
    sim.setObjectParent(s0,current_body,0)
    midp=(sim.getObjectPosition(Rj0,-1)[0]+sim.getObjectPosition(Rj1,-1)[0])/2
    sim.setObjectPosition(s0,-1,[midp-length_s+support_size[0],sim.getObjectPosition(Rj0,-1)[1]+0.025,sim.getObjectPosition(Rj0,-1)[2]-radius+support_size[0]/1.5])
    s0=[s0]
    int_wheels=math.floor(2*length_s/(support_size[0]))
    if int_wheels<=1:
        int_wheels=2
    for i in range(int_wheels-1):
        s1=(sim.copyPasteObjects([s0[-1]],0))
        sim.setObjectPosition(s1[0],s0[-1],[support_size[0],0,0])
        s0.append(s1[0])
    try:
        s1=sim.groupShapes(s0)
    except:
        s1=s0[0]
    s0 = sim.copyPasteObjects([s1],0)
    sim.setObjectPosition(s0[0],s1,[0.05,0.,0.])

    s2 = sim.copyPasteObjects([s0[0],s1],0)
    sim.setObjectPosition(s2[0],-1,[sim.getObjectPosition(s0[0],-1)[0],-sim.getObjectPosition(s0[0],-1)[1],sim.getObjectPosition(s0[0],-1)[2]])
    sim.setObjectPosition(s2[1],-1,[sim.getObjectPosition(s1,-1)[0],-sim.getObjectPosition(s1,-1)[1],sim.getObjectPosition(s1,-1)[2]])
    sim.setObjectParent(s0[0],current_body,0)
    sim.setObjectParent(s1,current_body,0)
    sim.setObjectParent(s2[0],current_body,0)
    sim.setObjectParent(s2[1],current_body,0)
    return Rj0, Lwheels[0], Rj1, Lwheels[1], radius

def build_links(sim,link_length,link_height,joint_length):
    link_dim=[[link_length,0.135,link_height],[0.015,0.135,0.025],[0.02,0.014,0.0255]]
    link_location=[0.,0.,-link_height/2.1-link_dim[1][2]/2.1]
    l0=sim.createPrimitiveShape(sim.primitiveshape_cuboid,link_dim[0])
    l1=sim.createPrimitiveShape(sim.primitiveshape_cuboid,link_dim[1])
    l2=[]
    for i in range(3):
        l2.append(sim.createPrimitiveShape(sim.primitiveshape_cuboid,link_dim[2]))
        
    sim.setObjectPosition(l1,l0,link_location)
    sim.setObjectPosition(l2[0],l0,[0.,0.0491,link_height/2.1+link_dim[2][2]/2.1])
    sim.setObjectPosition(l2[1],l0,[0.,-0.0491,link_height/2.1+link_dim[2][2]/2.1])
    sim.setObjectPosition(l2[2],l0,[0.,0.0,link_height/2.1+link_dim[2][2]/2.1])
    l2.append(l1)
    l2.append(l0)
    track_link=sim.groupShapes(l2)
    joint_link=sim.createJoint(sim.joint_revolute_subtype,sim.jointmode_dynamic,0,[0.055,0.015])
    sim.setObjectQuaternion(joint_link,joint_link,[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
    sim.setObjectPosition(joint_link,track_link,[0.0,0.034,0])
    sim.setObjectParent(joint_link,track_link,True)
    dummy=[]
    dummy.append(sim.createDummy(0.01))
    dummy.append(sim.createDummy(0.01))
    sim.setLinkDummy(dummy[0],dummy[1])
    sim.setObjectParent(dummy[0],track_link,True)
    sim.setObjectParent(dummy[1],track_link,True)
    sim.setObjectInt32Param(track_link,sim.shapeintparam_static,0)
    sim.setObjectInt32Param(track_link,sim.shapeintparam_respondable,1)    
    return track_link, joint_link, dummy

def build_track_wheels(sim,t0,t1,num_links_circ,radius):
    l0=sim.getObjectPosition(t0,sim.handle_world)
    l1=sim.getObjectPosition(t1,sim.handle_world)
    radius=abs(l0[2]-l1[2])/2
    wheel_base=abs(l0[0]-l1[0])

    w0=sim.createPrimitiveShape(sim.primitiveshape_cylinder,[2*radius,2*radius,0.02553])
    w1=sim.createPrimitiveShape(sim.primitiveshape_cylinder,[2*radius,2*radius,0.02553])
    sim.setObjectInt32Param(w0,sim.shapeintparam_static,0)
    sim.setObjectInt32Param(w0,sim.shapeintparam_respondable,1)
    sim.setObjectInt32Param(w1,sim.shapeintparam_static,0)
    sim.setObjectInt32Param(w1,sim.shapeintparam_respondable,1)
    sim.setObjectQuaternion(w0,w0,[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
    sim.setObjectQuaternion(w1,w1,[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
    sim.setObjectPosition(w0,t0,[radius,0,0.0491/2])
    sim.setObjectPosition(w1,t0,[radius,0,-0.0491/2])
    theta=np.linspace(0,2*math.pi,2*num_links_circ+1)
    rc=radius-0.005
    spokes=[]
    for i in range(2*num_links_circ):
        spokes.append(sim.createPrimitiveShape(sim.primitiveshape_cylinder,[0.01,0.01,0.015]))
        spokes.append(sim.createPrimitiveShape(sim.primitiveshape_cylinder,[0.01,0.01,0.015]))
        sim.setObjectQuaternion(spokes[-1],spokes[-1],[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
        sim.setObjectQuaternion(spokes[-2],spokes[-2],[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
        sim.setObjectPosition(spokes[-2],w0,[rc*math.cos(theta[i]),rc*math.sin(theta[i]),-0.015/2-0.02553/2])
        sim.setObjectPosition(spokes[-1],w1,[rc*math.cos(theta[i]),rc*math.sin(theta[i]),0.015/2+0.02553/2])
    spokes.append(w0)
    spokes.append(w1)
    wt=sim.groupShapes(spokes)
    j0=sim.createJoint(sim.joint_revolute_subtype,sim.jointmode_dynamic,0,[0.075,0.025])
    sim.setObjectQuaternion(j0,j0,[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
    sim.setObjectPosition(j0,wt,[0,0,0])
    sim.setObjectParent(wt,j0,True)
    wt2=sim.copyPasteObjects([j0,wt],0)
    j1=wt2[0]
    wt1=wt2[1]
    sim.setObjectPosition(j1,j0,[l1[0]-l0[0],0,0])
    return j0, j1, wt, wt1, l1[0]-l0[0], radius

def build_wheels(sim,radius,current_body,jlocation=[0,0,0],width=0.075):
    tread_height=1/39.37
    tread_length=1/39.37

    ## Build the wheel ##
    w0=sim.createPrimitiveShape(sim.primitiveshape_cylinder,[2*radius,2*radius,width])
    sim.setObjectInt32Param(w0,sim.shapeintparam_static,0)
    sim.setObjectInt32Param(w0,sim.shapeintparam_respondable,1)  

    sim.setObjectQuaternion(w0,w0,[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])

    ## Create the joint ##
    j0=sim.createJoint(sim.joint_revolute_subtype,sim.jointmode_dynamic,0,[0.075,0.025])
    sim.setObjectQuaternion(j0,j0,[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
    sim.setObjectPosition(j0,w0,[0,0,0])
    sim.setObjectParent(w0,j0,True)

    ## Place Treads on wheel ##
    num_treads=int(0.25*(2*(radius+tread_height/2)*math.pi/tread_length))
    theta=np.linspace(0,2*math.pi,num_treads+1)
    lt=tread_height/2+radius
    treads=[]
    for i in range(num_treads):
        treads.append(sim.createPrimitiveShape(sim.primitiveshape_cuboid,[tread_height,width,tread_length]))
        sim.setObjectQuaternion(treads[i],treads[i],[0.,-math.sin(theta[i]/2),0.,math.cos(theta[i]/2)])
        sim.setObjectPosition(treads[i],w0,[lt*math.cos(theta[i]),lt*math.sin(theta[i]),0])
    treads.append(w0)
    Rw=sim.groupShapes(treads)
    left_side=sim.copyPasteObjects([j0,Rw],0)

    sim.setObjectParent(j0,current_body,0)
    sim.setObjectPosition(j0,current_body,[jlocation[0],jlocation[1]/2,jlocation[2]])
    sim.setObjectParent(left_side[0],current_body,0)
    sim.setObjectPosition(left_side[0],current_body,[jlocation[0],-jlocation[1]/2,jlocation[2]])   
    set_joint_mode(sim,[j0,left_side[0]],j_spring=[],jtype='velocity')
    ## Set the wheels relative to the body tomorrow ##
    return j0, left_side[0]

def build_planet_wheels(sim,radius,current_body,jlocation=[0,0,0],joint_type='fixed',num_planets=3,width=0.075,pwheel_radius=2/39.37):
    arm_width=0.005
    ## Build the center wheel ##
    w0=sim.createPrimitiveShape(sim.primitiveshape_cylinder,[radius*0.75,radius*0.75,width])

    sim.setObjectQuaternion(w0,w0,[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
    j0=sim.createJoint(sim.joint_revolute_subtype,sim.jointmode_dynamic,0,[0.075,0.025])
    sim.setObjectQuaternion(j0,j0,[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
    sim.setObjectPosition(j0,w0,[0,0,0])
    
    ## Build the planet wheels ##
    theta=np.linspace(0,2*math.pi,num_planets+1)
    planet_ids=[]
    links=[]
    joint_ids=[]
    for i in range(num_planets):
        ## Create the planet wheels ##
        planet_ids.append(sim.createPrimitiveShape(sim.primitiveshape_cylinder,[pwheel_radius,pwheel_radius,width]))
        sim.setObjectQuaternion(planet_ids[-1],planet_ids[-1],[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
        sim.setObjectPosition(planet_ids[-1],w0,[radius*math.cos(theta[i]),radius*math.sin(theta[i]),0])

        ## Add joints to the planet wheels ## (Removed this until I know how I want to control the joints)
        # joint_ids.append(sim.createJoint(sim.joint_revolute_subtype,sim.jointmode_dynamic,0,[0.075,0.025]))
        # sim.setObjectQuaternion(joint_ids[-1],joint_ids[-1],[math.sin(math.pi/4),0,0,math.cos(math.pi/4)])
        # sim.setObjectPosition(joint_ids[-1],planet_ids[-1],[0,0,0])
        # sim.setObjectParent(planet_ids[-1],joint_ids[-1],0)
        # sim.setJointMode(joint_ids[i],sim.jointmode_dynamic,0)
        # if joint_type=='fixed':
        #     vdes=0
        # else:
        #     vdes=100/60*360
        # sim.setObjectInt32Param(joint_ids[-1],sim.jointintparam_dynctrlmode,sim.jointdynctrl_velocity)    
        # sim.setJointTargetVelocity(joint_ids[i],vdes,[-1000,10000])

        ## Build the arms onto planet ##
        links.append(sim.createPrimitiveShape(sim.primitiveshape_cuboid,[radius,arm_width,0.5*pwheel_radius]))
        sim.setObjectQuaternion(links[-1],links[-1],[0.,math.sin(-theta[i]/2),0,math.cos(theta[i]/2)])
        sim.setObjectPosition(links[-1],w0,[radius/2*math.cos(theta[i]),radius/2*math.sin(theta[i]),width/1.9+arm_width])
        links.append(sim.createPrimitiveShape(sim.primitiveshape_cuboid,[radius,arm_width,0.5*pwheel_radius]))
        sim.setObjectQuaternion(links[-1],links[-1],[0.,math.sin(-theta[i]/2),0,math.cos(theta[i]/2)])
        sim.setObjectPosition(links[-1],w0,[radius/2*math.cos(theta[i]),radius/2*math.sin(theta[i]),-(width/1.9+arm_width)])
    links.append(w0)

    p0=sim.groupShapes(links+planet_ids)
    sim.setObjectParent(p0,j0,0)
    sim.setObjectInt32Param(p0,sim.shapeintparam_static,0)
    sim.setObjectInt32Param(p0,sim.shapeintparam_respondable,1)    

    lw=sim.copyPasteObjects([p0,j0],0)

    sim.setObjectParent(j0,current_body,0)
    sim.setObjectPosition(j0,current_body,[jlocation[0],jlocation[1]/2,jlocation[2]])
    sim.setObjectParent(lw[1],current_body,0)
    sim.setObjectPosition(lw[1],current_body,[jlocation[0],-jlocation[1]/2,jlocation[2]])
    set_joint_mode(sim,[j0,lw[1]],j_spring=[],jtype='velocity')
    return j0, lw[1]

def set_joint_mode(sim,j0,j_spring=[],jtype='force',max_torque=150.):
    for i in j0:
        if jtype=='force':
            sim.setObjectInt32Param(i,sim.jointintparam_dynctrlmode,sim.jointdynctrl_force)    
        elif jtype=='velocity':
            sim.setObjectInt32Param(i,sim.jointintparam_dynctrlmode,sim.jointdynctrl_velocity)  
        else:
            sim.setObjectInt32Param(i,sim.jointintparam_dynctrlmode,sim.jointdynctrl_spring)
            sim.setObjectFloatParam(i,sim.jointfloatparam_kc_k,j_spring[0])
            sim.setObjectFloatParam(i,sim.jointfloatparam_kc_c,j_spring[1])
        sim.setJointTargetForce(i,max_torque)

def set_body_joints(sim,location,parent,spring_coeff=[],orientation=[0,0,0]):
    joint_link=sim.createJoint(sim.joint_revolute_subtype,sim.jointmode_dynamic,0,[0.055,0.015])
    set_joint_mode(sim,[joint_link],j_spring=spring_coeff,jtype='spring')
    sim.setObjectParent(joint_link,parent,0)
    sim.setObjectPosition(joint_link,parent,location)
    sim.setObjectQuaternion(joint_link,joint_link,[orientation[0],orientation[1],orientation[2],math.cos(math.pi/4)])

    return joint_link

def build_vehicles(sim,nodes):
    lead_bodyid=0
    props=[]
    joint_location=[0,0,0]
    prev_track=False
    err=False
    i=0
    while err==False:
        if i==0:
            current_body=generate_body(sim,[nodes[i]['length'],nodes[i]['width'],nodes[i]['height']])
            lead_bodyid=copy.copy(current_body)
            sim.setObjectPosition(current_body,current_body,[0,0,0.5])
        else:
            current_body=generate_body(sim,[nodes[i]['length'],nodes[i]['width'],nodes[i]['height']])
            sim.setObjectParent(current_body,current_joint,0)
            
            pos=JRot.as_matrix()@np.array([1/39.37+nodes[i]['length']/2,-joint_location[1],-joint_location[2]])
            sim.setObjectPosition(current_body,current_joint,[pos[0],pos[1],pos[2]])


        if len(nodes[i]['childern'])==2:
            if nodes[i+1]['type']=='planet wheel':
                current_props=build_planet_wheels(sim,nodes[i+1]['radius'],current_body,nodes[i+1]['location'])
                props.append(current_props[:2])
            elif nodes[i+1]['type']=='wheel':
                current_props=build_wheels(sim,nodes[i+1]['radius'],current_body,nodes[i+1]['location'])  
                props.append(current_props[:2])    
            else:
                if prev_track==False:
                    wheel_base=abs((nodes[i+2]['location'][0]+1/39.37+nodes[i+3]['length']/2+nodes[i+4]['location'][0])-nodes[i+1]['location'][0])
                    current_props=generate_tracks(sim,nodes[i+1]['radius'],wheel_base,current_body,nodes[i+1]['location'])
                    props.append(current_props[:2])
                    prev_track=True
                else:
                    sim.setObjectParent(current_props[0],current_body,0)
                    sim.setObjectParent(current_props[1],current_body,0)
                    prev_track=False
            joint_node=copy.copy(nodes[i+2])
            
            i+=3
        else:
            joint_node=copy.copy(nodes[i+1])
            i+=2

        if i>=len(nodes)-1:
            err=True
        else:
            current_joint=set_body_joints(sim,joint_node['location'],current_body,spring_coeff=joint_node['active'],orientation=joint_node['orientation'])
            joint_location=copy.copy(joint_node['location'])   
            JRot=R.from_quat([-joint_node['orientation'][0],-joint_node['orientation'][1],-joint_node['orientation'][2],math.cos(math.pi/4)])        
            # JRot=R.from_quat([-joint_node['orientation'][0],-joint_node['orientation'][1],-joint_node['orientation'][2],joint_node['orientation'][3]])    
        
        # vrep_nodes.append(generate_body(sim,[nod['length'],nod['width'],nod['height']]))
    x_current, edge_current = convert2tensor(nodes)
    return props, lead_bodyid, x_current, edge_current, nodes

def build_steps(sim,num_steps=50,step_height=6.5/39.37,slope=30):
    b0=[]
    for i in range(num_steps):
        b0.append(sim.createPrimitiveShape(sim.primitiveshape_cuboid,[1.,4.,step_height]))
        l=step_height/math.tan(slope*math.pi/180)
        sim.setObjectPosition(b0[-1],b0[-1],[-l*i-1.0,0,step_height/2+step_height*i])

    final_pos=[(-l*(i+1)-0.5),step_height*i]
    Rw=sim.groupShapes(b0)
    # sim.setObjectInt32Param(Rw,sim.shapeintparam_static,0)
    sim.setObjectInt32Param(Rw,sim.shapeintparam_respondable,1)
    sim.cameraFitToView(0)
    return final_pos, step_height, slope, Rw
    
def convert2tensor(nodes):
    ## Save as a datalist ##
    # data=[]
    # edge_attribute=torch.tensor
    # edge_index=torch.tensor
    # x=torch.tensor
    # count=0
    type=['wheel','planet wheel','track']
    x=[]
    edge_index=[]
    for i, nod in enumerate(nodes):
        if nod['name']=='body':
            x.append([nod['length'],nod['width'],nod['height'],0.,0.])
            for j in (nod['childern']):
                edge_index.append([i,j])
            if i>0:
                edge_index.append([nod['parents'],i])
        elif nod['name']=='prop':
            for j, prop_type in enumerate(type):
                if prop_type==nod['type']:
                    prop_index=j 
            x.append([prop_index,nod['radius'],nod['location'][0],nod['location'][1],nod['location'][2]])
        else:
            for j, ori in enumerate(nod['orientation']):
                if ori!=0:
                    ori_index=j             
            x.append([nod['active'][0],nod['active'][1],ori_index,nod['location'][0],nod['location'][2]])
        
    return x, edge_index

def create_vehicles(x_reals,x_ints,num_bodies=4,num_body_reals=3,num_prop_reals=4,num_joint_reals=4,num_prop_ints=4,num_joint_ints=3):
    ## Determine the number of bodies ##
    bodies=np.argmax(x_ints[:3])+2
    body_reals=np.reshape(x_reals[:num_bodies*num_body_reals],(num_bodies,num_body_reals))
    prop_reals=np.reshape(x_reals[num_bodies*num_body_reals:num_bodies*(num_body_reals+num_prop_reals)],(num_bodies,num_prop_reals))
    joint_reals=np.reshape(x_reals[num_bodies*(num_body_reals+num_prop_reals):],(num_bodies-1,num_joint_reals))
    prop_ints=np.reshape(x_ints[3:3+num_prop_ints*num_bodies],(num_bodies,num_prop_ints))
    joint_ints=np.reshape(x_ints[3+num_prop_ints*num_bodies:],(num_bodies-1,num_joint_ints))
    joint_ints=np.vstack((joint_ints,np.array([0,0,1])))
    joint_reals=np.vstack((joint_reals,np.zeros(num_joint_reals)))
    nodes=[]
    edges=[]
    body_id=0
    prop_types=['none','wheel','planet wheel','track']
    for i in range(bodies):
        # self.body_nodes={"name":"body","location":[],"length": 0,"width":20/39.37,"height":12/39.37,"clearance":0,"childern": [],"parents": [],"index":0}
        # self.joint_nodes={"name":"joint","location": [0, 0, 0],"orientation":[0,0,0],"active":[],"childern": [],"parents": [],"index":1}
        # self.prop_nodes={"name":"prop","location": [0, 0, 0],"radius":0,"childern": [],"parents": [],"type":'none'}
        propid=np.argmax(prop_ints[i,:]).item()
        jointid=np.argmax(joint_ints[i,:])
        
        # nodes.append([body_reals[i,0],body_reals[i,1],body_reals[i,1],0.,0.])
        if propid!=0:
            # nodes.append([propid-1, prop_reals[i,0], prop_reals[i,1], prop_reals[i,2], prop_reals[i,3]])
            ## Determine what type of mechanism is next [0=none, 1=wheel, 2=planet wheel, 3=track] ##
            # kinda cheating for right now #
            nodes.append({"name":"body","location":[],"length": body_reals[i,0].item(),"width":0.508001,"height":0.3048006,"clearance":0,"childern": [body_id+1,body_id+2],"parents": [],"index":0})
            nodes.append({"name":"prop","location": [prop_reals[i,1].item(), 0.658001006, -0.102400303],"radius":prop_reals[i,0].item(),"childern": [],"parents": [],"type":prop_types[propid]})
            # This is the actual NN recreation #
            # nodes.append({"name":"body","location":[],"length": body_reals[i,0].item(),"width":body_reals[i,1].item(),"height":body_reals[i,2].item(),"clearance":0,"childern": [body_id+1,body_id+2],"parents": [],"index":0})
            # nodes.append({"name":"prop","location": [prop_reals[i,1].item(), prop_reals[i,2].item(), prop_reals[i,3].item()],"radius":prop_reals[i,0].item(),"childern": [],"parents": [],"type":prop_types[propid]})
            edges.append([body_id,body_id+1])
            edges.append([body_id,body_id+2])
            index_increase=3
        else:
            nodes.append({"name":"body","location":[],"length": body_reals[i,0].item(),"width":body_reals[i,1].item(),"height":body_reals[i,2].item(),"clearance":0,"childern": [body_id+1],"parents": [],"index":0})
            edges.append([body_id,body_id+1])
            index_increase=2
        jori=[0.,0.,0.]
        jori[jointid]=math.sin(math.pi/4)
        nodes.append({"name":"joint","location": [joint_reals[i,2].item(), 0, joint_reals[i,3].item()],"orientation":jori,"active":[joint_reals[i,0].item(),joint_reals[i,1].item()],"childern": [],"parents": [],"index":1})
        # nodes.append([joint_reals[i,0],joint_reals[i,1],jointid,joint_reals[i,2],joint_reals[i,3]])
        body_id+=index_increase
    return nodes, edges