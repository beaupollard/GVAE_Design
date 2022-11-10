import numpy as np
import math

def get_gauss_rand(in_mean,in_std=0,l_lim=-1000000,u_lim=1000000):
    outp = l_lim-1
    while outp<l_lim or outp>u_lim:
        outp = np.random.normal(in_mean,in_std)
    return outp

def generate_body(sim,body_size=[0.3,0.15,0.2]):
    density=50 #lbs/ft^3
    volume=body_size[0]*body_size[1]*(body_size[2]+2/39.37)
    b0=sim.createPrimitiveShape(sim.primitiveshape_cuboid,body_size)
    sim.setShapeMass(b0,volume*density)
    return b0

def generate_tracks(sim,radius, wheel_base):
    
    link_length=0.055
    link_height=0.009/2.
    joint_length=0.034
    track_links=[]
    track_joints=[]
    tl, tj, dummy_ids = build_links(sim,link_length,link_height*2,joint_length)
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
    linkandjoint_width=2*joint_length#link_width/2+joint_length
    num_tracks_circ=math.ceil(math.pi*(radius+link_height)/linkandjoint_width)
    theta=math.pi/num_tracks_circ
    for i in range(num_tracks_circ):
        sim.setObjectQuaternion(track_joints[i],track_joints[i],[0,0,math.sin(theta/2),math.cos(theta/2)])
    i+=int((num_tracks-2*num_tracks_circ)/2)+1

    for j in range(num_tracks_circ):
        sim.setObjectQuaternion(track_joints[i+j],track_joints[i+j],[0,0,math.sin(theta/2),math.cos(theta/2)])
    
    sim.setObjectParent(dummy_ids[-1],track_joints[-1],True)
    Rj0, Rj1, Rw0, Rw1 = build_track_wheels(sim,track_links[0],track_links[i],num_tracks_circ)
    Ltrack_links=sim.copyPasteObjects(track_links+track_joints+dummy_ids,0)

    Lwheels = sim.copyPasteObjects([Rj0,Rj1,Rw0,Rw1],0)
    return track_links[0], Rj0, Rj1, Ltrack_links[0], Lwheels[0], Lwheels[2]

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
    return track_link, joint_link, dummy

def build_track_wheels(sim,t0,t1,num_links_circ):
    l0=sim.getObjectPosition(t0,sim.handle_world)
    l1=sim.getObjectPosition(t1,sim.handle_world)
    radius=abs(l0[2]-l1[2])/2
    wheel_base=abs(l0[0]-l1[0])

    w0=sim.createPrimitiveShape(sim.primitiveshape_cylinder,[2*radius,2*radius,0.02553])
    w1=sim.createPrimitiveShape(sim.primitiveshape_cylinder,[2*radius,2*radius,0.02553])
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
    return j0, j1, wt, wt1

def build_wheels(sim,radius,width=0.075):
    tread_height=3/39.37
    tread_length=2/39.37

    ## Build the wheel ##
    w0=sim.createPrimitiveShape(sim.primitiveshape_cylinder,[2*radius,2*radius,width])
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
    return j0, Rw, left_side[0], left_side[1]

def build_planet_wheels(sim,radius,joint_type='fixed',num_planets=3,width=0.075,pwheel_radius=2/39.37):
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

    lw=sim.copyPasteObjects([p0,j0],0)
    return j0, p0, lw[0], lw[1]

def set_joint_mode(sim,j0,j_type='force'):
    for i in j0:
        if j_type=='force':
            sim.setObjectInt32Param(j0,sim.jointintparam_dynctrlmode,sim.jointdynctrl_force)    
        else:
            sim.setObjectInt32Param(j0,sim.jointintparam_dynctrlmode,sim.jointdynctrl_spring)
