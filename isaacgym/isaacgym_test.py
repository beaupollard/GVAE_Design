from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np
from mppi_ctrl import MPPI


# initialize gym
gym = gymapi.acquire_gym()

sim_parms = gymapi.SimParams()
sim_parms.dt = 0.01
sim_parms.physx.use_gpu = True
sim_parms.use_gpu_pipeline = True

sim = gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_parms)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.distance = 0.5
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "../assets"
asset_file = 'robot2.urdf'

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link =False
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# set up the env grid
num_envs = int(2**11)
actors_per_env = 1
dofs_per_actor = 11
num_per_row = 8
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(17.2, 2.0, 16)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

gym.prepare_sim(sim)
gym.simulate(sim)

# ## Aquire tensor descriptors #
root_states_desc = gym.acquire_actor_root_state_tensor(sim)
dof_states_desc = gym.acquire_dof_state_tensor(sim)
# force_desc = gym.acquire_dof_force_tensor(sim)

# ## Pytorch interop ##
root_states = gymtorch.wrap_tensor(root_states_desc)
dof_states = gymtorch.wrap_tensor(dof_states_desc)
# force_states = gymtorch.wrap_tensor(force_desc)

# ## View information as a vector of envs ##
root_states_vec = root_states.view(num_envs,actors_per_env,13)
dof_states_vec = dof_states.view(num_envs,int(actors_per_env*dof_states.size()[0]/num_envs),2)


## Initialize MPPI Controller ##
active_index=[0,1,2,3]
ctrl=MPPI(sim,gym,num_dofs,num_envs,20,active_index,[1.0,200.,0.01],[4],[0])
count=0
while True:
    # step the sim #
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh state tensor #
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    # gym.refresh_dof_force_tensor(sim)

    # determine next command input #
    if count>55:
        ctrl.get_command(root_states,dof_states)

    # refresh state tensor #
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    # gym.refresh_dof_force_tensor(sim)

    # # apply controls
    # forces = 2.0*torch.rand((num_envs,5),dtype=torch.float32, device="cuda:0")

    # # upwrap tensor
    # forces_desc = gymtorch.unwrap_tensor(forces)

    # # apply forces
    # gym.set_dof_actuation_force_tensor(sim,forces_desc)

    # # use PD controls
    # gym.set_dof_position_target_tensor(sim,pos_targets_desc)
    # gym.set_dof_velocity_target_tensor(sim,vel_targets_desc)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    count+=1
