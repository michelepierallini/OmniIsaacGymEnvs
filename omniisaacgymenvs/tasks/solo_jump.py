from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.solo import Solo
from omniisaacgymenvs.robots.articulations.views.solo_view import SoloView
from omniisaacgymenvs.tasks.utils.terrain_jump import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.simulation_context import SimulationContext

import numpy as np
import torch
# import math
# import time
# import csv


from pxr import UsdPhysics, UsdLux


class SoloJumpTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.height_samples = None
        self.custom_origins = False
        self.init_done = False

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}   
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["fallen_over"] = self._task_cfg["env"]["learn"]["fallenOverRewardScale"]
        self.rew_scales["hasnt_flew"] = self._task_cfg["env"]["learn"]["hasntFlewRewardScale"]
        self.rew_scales["trajectory"] = self._task_cfg["env"]["learn"]["trajectoryRewardScale"]
        self.rew_scales["storage_height"] = self._task_cfg["env"]["learn"]["storageheightRewardScale"]   
        self.rew_scales["velocity_storage"] = self._task_cfg["env"]["learn"]["storagevelocityRewardScale"]      
        self.rew_scales["velocity_final"] = self._task_cfg["env"]["learn"]["velocityfinalReward"]   
        self.rew_scales["fallen_over_final"] = self._task_cfg["env"]["learn"]["fallenOverFinalRewardScale"]       
        self.rew_scales["roll"] = self._task_cfg["env"]["learn"]["rollRewardScale"]
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["TerminationRewardScale"]

        self.rew_scales["velocity_x"] = self._task_cfg["env"]["learn"]["velocityXRewardScale"]
        self.rew_scales["velocity_z"] = self._task_cfg["env"]["learn"]["velocityZRewardScale"]
        self.rew_scales["time"] = self._task_cfg["env"]["learn"]["timeRewardScale"]
        self.rew_scales["ground_forces"] = self._task_cfg["env"]["learn"]["ground_forcesRewardScale"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.first_height = self._task_cfg["env"]["terrain"]["first_height"]
        self.base_threshold = 0.2
        self.knee_threshold = 0.1
        self.max_ground_time_s = 2.5 #sec   0.8
        self.max_ground_time = int(self.max_ground_time_s/ self.dt + 0.5)  # self.dt
        self.flying_threshold = 1 # meters
        self.robot_default_height = 0.66
        self.base_init_state[2] += self.robot_default_height * 1.1
        self._device = 'cuda:0'
        self.count = 0
        # with open(Evolution_Solo_Jump, 'w', newline="") as file:
        #    writer = csv.writer(file)


        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self.first_landing = torch.zeros(self._num_envs, dtype=torch.int, device=self._device)
        self.first_takeoff = torch.zeros(self._num_envs, dtype=torch.int, device=self._device)
        self.not_evaluated = torch.zeros(self._num_envs, dtype=torch.int, device=self._device)
        self.has_early_landed = torch.zeros(self._num_envs, dtype=torch.int, device=self._device)
        self.first_landing_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.jump_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._num_observations = 187
        self._num_actions = 12

        self.gravity = torch.tensor(self._task_cfg["sim"]["gravity"][2], device=self.device)

        self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"]["staticFriction"]
        self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"]["dynamicFriction"]
        self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"]["restitution"]
   
        self._task_cfg["sim"]["add_ground_plane"] = False
        self._env_spacing = 0.0

        RLTask.__init__(self, name, env)

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # initialize some data used later on
        self.up_axis_idx = 2
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self._task_cfg)
        self.commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False) # distance, yaw, height, type terrain
        self.gravity_vec = torch.tensor(get_axis_params(-1., self.up_axis_idx), dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.last_base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)

        self.height_points = self.init_height_points()
        self.measured_heights = None
        # joint positions offsets
        self.default_dof_pos = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)
        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = { "distance": torch_zeros(), "velocity_final": torch_zeros(), "velocity_storage": torch_zeros(), "height": torch_zeros(), "velocity_x": torch_zeros(), "velocity_z": torch_zeros(), "time": torch_zeros(), "ground_force": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(), "action_rate": torch_zeros(), "roll": torch_zeros(), "trajectory": torch_zeros()}
        return

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:11] = 0. # commands
        noise_vec[11:23] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[23:35] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[35:175] = self._task_cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        noise_vec[175:187] = 0. # previous actions
        return noise_vec
    
    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False) # 10-50cm on each side
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False) # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _create_trimesh(self):
        self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size , -self.terrain.border_size , 0.0])
        add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)  
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_terrain()
        self.get_solo()
        super().set_up_scene(scene)
        self._solos = SoloView(prim_paths_expr="/World/envs/.*/solo", name="solo_view", track_contact_forces=True)
        scene.add(self._solos)
        scene.add(self._solos._knees)
        scene.add(self._solos._base)
        scene.add(self._solos._feet)

    def get_terrain(self):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if not self.curriculum: self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1

        self.terrain_levels = torch.randint(0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
 

        self.terrain_types = torch.randint(0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)

        self.commands[:, 0] = self.first_height + self.terrain_levels * 0.3
        self.commands[:, 1] = self.terrain_types

        self._create_trimesh()
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            
    def get_solo(self):
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False)

        solo_translation = self.base_init_state[:3]  # torch.tensor([0.0, 0.0, 0.0])  # self.solo_init_pos
        solo_orientation = self.base_init_state[3:7]  # torch.tensor([1.0, 0.0, 0.0, 0.0])  # self.solo_init_orientation

        self.random_position = torch.zeros(self._num_envs, device=self._device)
        self.trajectory = torch.zeros(self._num_envs, 3, 20, device=self._device)
        self.base_lin_target = torch.zeros(self.num_envs, 3, device=self._device)

        solo = Solo(prim_path=self.default_zero_env_path + "/solo",
                    name="solo",
                    translation=solo_translation,
                    orientation=solo_orientation)
        self._sim_config.apply_articulation_settings("solo", get_prim_at_path(solo.prim_path), self._sim_config.parse_actor_config("solo"))
        solo.set_solo_properties(self._stage, solo.prim)
        solo.prepare_contacts(self._stage, solo.prim)

        self.dof_names = solo.dof_names
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def post_reset(self):
        for i in range(self.num_envs):
            self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
        self.num_dof = self._solos.num_dof
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        self.knee_pos = torch.zeros((self.num_envs*4, 3), dtype=torch.float, device=self.device)
        self.knee_quat = torch.zeros((self.num_envs*4, 4), dtype=torch.float, device=self.device)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        self.init_done = True

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        self.update_terrain_level(env_ids)
        self.base_pos[env_ids] = self.base_init_state[0:3]
        self.base_pos[env_ids, 0:2] += self.env_origins[env_ids, 0:2]
        self.base_pos[env_ids, 0] += -2   # later on da eliminare


        # later on
        # self.random_position[env_ids] = torch_rand_float(-2.5, -2, (len(env_ids), 1), device=self.device).squeeze(1)  
        # self.base_pos[env_ids, 0] += self.random_position[env_ids].squeeze().t()
        # self.base_quat[env_ids] = self.base_init_state[3:7]
        # angle = torch_rand_float(0, 2*3.14, (len(env_ids), 1), device=self.device) / 2
        # self.base_quat[env_ids, 0:1] = torch.cos(angle)
        # self.base_quat[env_ids, 3:4] = torch.sin(angle)

        self.base_quat[env_ids] = self.base_init_state[3:7]


        # self.commands[env_ids, 0:1] = self.base_pos[env_ids, 0:1] - self.env_origins[env_ids, 0:1]
        # self.commands[env_ids, 1:2] = 2 * torch.atan2(self.base_quat[env_ids, 3:4], self.base_quat[env_ids, 0:1]) 

        self.create_trajectory(self.commands[:, 0], env_ids)

        self.base_velocities[env_ids] = self.base_init_state[7:]

        self._solos.set_world_poses(positions=self.base_pos[env_ids].clone(),
                                    orientations=self.base_quat[env_ids].clone(),
                                    indices=indices)
        self._solos.set_velocities(velocities=self.base_velocities[env_ids].clone(),
                                   indices=indices)
        self._solos.set_joint_positions(positions=self.dof_pos[env_ids].clone(),
                                        indices=indices)
        self._solos.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(),
                                         indices=indices)

        # self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.

        self.progress_buf[env_ids] = 0.0
        self.jump_time[env_ids] = 0.0
        self.first_landing_time[env_ids] = 0.0
        self.first_landing[env_ids] = 0
        self.first_takeoff[env_ids] = 0
        self.has_early_landed[env_ids] = 0

        # self.reset_buf[env_ids] = 0
        self.last_base_pos [env_ids] = 0.
        self.last_base_quat [env_ids] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
    
    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # do not change on initial reset
            return
        # root_pos, _ = self._solos.get_world_poses(clone=False)

        self.terrain_levels -= 1 * torch.logical_or(self.has_fallen == True , self.hasnt_flew == True)
        self.terrain_levels += 1 * (self.mission_complete == True)

        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def refresh_dof_state_tensors(self):
        self.dof_pos = self._solos.get_joint_positions(clone=False)
        self.dof_vel = self._solos.get_joint_velocities(clone=False)
    
    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._solos.get_world_poses(clone=False)
        self.base_velocities = self._solos.get_velocities(clone=False)
        self.knee_pos, self.knee_quat = self._solos._knees.get_world_poses(clone=False)

    def pre_physics_step(self, actions):
        if not self._env._world.is_playing():
            return

        self.actions = actions.clone().to(self.device)
        for i in range(self.decimation):
            if self._env._world.is_playing():
                torques = torch.clip(self.Kp*(self.action_scale*self.actions + self.default_dof_pos - self.dof_pos) - self.Kd*self.dof_vel, -80., 80.)
                self._solos.set_joint_efforts(torques)
                self.torques = torques
                SimulationContext.step(self._env._world, render=False)
                self.refresh_dof_state_tensors()
    
    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self._env._world.is_playing():

            self.refresh_dof_state_tensors()
            self.refresh_body_state_tensors()

            self.common_step_counter += 1
            if self.common_step_counter % self.push_interval == 0:
                self.push_robots()

            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

            self.check_termination()
            self.get_states()

            self.single_metric()

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            
            if len(env_ids) > 0:               
                self.reset_idx(env_ids)

            self.get_observations()

            if self.obs_buf.isnan().any():
                print(self.obs_buf.isnan().any(dim=0).nonzero(as_tuple=False).flatten()) 


            if self.add_noise:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

            self.last_actions[:] = self.actions[:]
            self.last_dof_vel[:] = self.dof_vel[:]

            self.count += 1

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    

    def single_metric (self):

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"] 

        # torque penalty  
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]  

        # fallen over penalty
        rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        finish = self.env_origins[:, 0:2]
        # finish[:, 2] += self.robot_default_height

        rew_distance = torch.norm(self.base_pos[:, 0:2] - self.env_origins[:, 0:2], dim = -1) * self.rew_scales["roll"]

        rew_mission_complete = self.mission_complete * self.rew_scales["termination"]
       
        # total reward
        self.rew_buf = rew_joint_acc + rew_torque + rew_fallen_over + rew_action_rate + rew_distance + rew_mission_complete

        # log episode reward sums
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["distance"] += rew_distance
        

    def general_metrics (self):
          
            self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self.condition_start = (self.first_landing == 0) & (self.base_lin_vel[:, 2] < 0.1) 
            self.condition_invert = (abs(self.base_lin_vel[:, 2]) < 0.15) & (self.base_pos[:, 2] < self.robot_default_height*0.75) & (self.first_takeoff == 0)
            self.condition_first_landing = (torch.norm(self._solos.get_contact_forces()[0].reshape(self._num_envs, 4, 3), dim=-1) > 0.1).any(dim=-1) & (self.first_landing == 0).t()
            self.condition_ground = (torch.norm(self._solos.get_contact_forces()[0].reshape(self._num_envs, 4, 3), dim=-1) > 0.1).any(dim=-1) & (self.first_landing == 1).t()
            self.condition_first_takeoff = ((self.first_landing == 1) & (self.first_takeoff == 0) & (torch.norm(self._solos.get_contact_forces()[0].reshape(self._num_envs, 4, 3), dim=-1) < 0.1).all(dim=-1).t())
            self.condition_fly = (torch.norm(self._solos.get_contact_forces()[0].reshape(self._num_envs, 4, 3), dim=-1) < 0.1).all(dim=-1) & (self.first_takeoff == 1).t()
            self.condition_second_landing = (torch.norm(self._solos.get_contact_forces()[0].reshape(self._num_envs, 4, 3), dim=-1) > 0.1).any(dim=-1) & (self.first_landing == 1).t() & (self.base_pos[:, 2] > self.commands[:, 0]+self.robot_default_height*0.75)
            self.condition_early_landing = (torch.norm(self._solos.get_contact_forces()[0].reshape(self._num_envs, 4, 3), dim=-1) > 0.1).any(dim=-1) & (self.first_landing == 1).t() & (self.base_pos[:, 2] < self.commands[:, 0]+self.robot_default_height*0.75)

            self.how_many_0 = int(torch.sum(self.condition_start).item() / self._num_envs * 1024)
            self.how_many_1 = int(torch.sum(self.condition_ground).item() / self._num_envs * 1024)
            self.how_many_2 = int(torch.sum(self.condition_fly).item() / self._num_envs * 1024)
            self.how_many_3 = int(torch.sum(self.condition_second_landing).item() / self._num_envs * 1024)
            self.how_many_fall = int(torch.sum(self.has_fallen).item() / self._num_envs * 1024)
            self.how_many_no_flew = int(torch.sum(self.hasnt_flew).item() / self._num_envs * 1024)
            self.how_many_horphans = int(torch.sum(self.horphans).item() / self._num_envs * 1024)

            # row = [self.how_many_0, self.how_many_1, self.how_many_2, self.how_many_3, self.how_many_fall, self.how_many_no_flew]
            # writer.writerrow(row)

            if self.count%12 == 0:
                print("HOW MANY", self.how_many_0, self.how_many_1, self.how_many_2, self.how_many_3, self.how_many_fall, self.how_many_no_flew, "out of 1024")
                # print(torch.norm(self._solos.get_contact_forces()[0].reshape(self._num_envs, 4, 3), dim=-1))

            self.zero_rewards()

            self.first_landing_fun()

            if any(self.condition_ground):
                
               self.calculate_metrics_first()

            self.first_takeoff_fun()

            if any(self.condition_fly):
                self.calculate_metrics_second()

            if any(self.condition_second_landing):
                self.calculate_metrics_third()

            self.early_landing()

            self.rew_buf = torch.clip(self.rew_buf , min=-2, max=None)

            self.conditions = self.condition_start | self.condition_ground | self.condition_fly | self.condition_second_landing
            self.not_evaluated = ~self.conditions
            
            # print(torch.sum(not_evaluated).item())

            # print((torch.norm(self._solos.get_contact_forces()[0].reshape(self._num_envs, 4, 3), dim=-1) > 0.1).any(dim=-1)[not_evaluated]) # quasi tutti toccano
            # print(self.base_lin_vel[not_evaluated, 2])  # quasi tutti > 0
            # print(self.first_landing[not_evaluated])  # quasi tutti 0
            # print(self.base_pos[not_evaluated, 2]) # poco sotto default
    

    def push_robots(self):
        self.base_velocities[:, 0:2] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device) # lin vel x/y
        self._solos.set_velocities(self.base_velocities)
    
    def check_termination(self):
 
        knee_contact = torch.norm(self._solos._knees.get_net_contact_forces(clone=False).view(self._num_envs, 4, 3), dim=-1) > 1.       
        # self.has_fallen = (torch.norm(self._solos._base.get_net_contact_forces(clone=False), dim=-1) > 0.1) | (torch.sum(knee_contact, dim=-1) > 0.1)
        self.has_fallen = torch.zeros_like(self.reset_buf)

        # self.hasnt_flew = (self.progress_buf >= self.max_ground_time) & ((self.first_takeoff == 0) | ~self._solos.is_base_below_threshold(0.8, 0))  # no for single_metric

        # self.mission_complete = ((torch.abs(self.base_pos[:, 2:3] - (self.commands[:, 0] + self.robot_default_height)) < 0.15).t() & (abs(self.base_lin_vel) < 0.2).all(dim=1)).flatten() & (torch.norm(self._solos.get_contact_forces()[0].reshape(self._num_envs, 4, 3), dim=-1) > 0.1).all(dim=-1)# if vel low and heigth position reached
        self.mission_complete = torch.zeros_like(self.reset_buf)

        # self.reset_buf = (self.has_fallen | self.hasnt_flew | self.mission_complete).long()
        self.reset_buf = (self.has_fallen | self.mission_complete).long()

        if self.mission_complete.any():
            print("MISSION COMPLETED")
            print("POSITION", self.base_pos[self.mission_complete, :].numpy())
            print("VELOCITY", self.base_lin_vel[self.mission_complete, :].numpy())
            print("LIFETIME", (self.progress_buf[self.mission_complete] * self.dt ).numpy())

        self.horphans = self.not_evaluated 
        # self.reset_buf += self.horphans.clone().int()

    def zero_rewards(self):
        index = self.condition_start

        # keep going down until a point defined as a jump_energy_storage    
        rew_height = self.base_pos [index, 2] * self.rew_scales["storage_height"]

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel[index, :] - self.dof_vel[index, :]), dim=1) * self.rew_scales["joint_acc"] 

        # torque penalty  
        rew_torque = torch.sum(torch.square(self.torques[index, :]), dim=1) * self.rew_scales["torque"]  

        # fallen over penalty
        rew_fallen_over = self.has_fallen[index] * self.rew_scales["fallen_over"]

        # roll penalty           
        rew_roll = torch.square(self.base_ang_vel[index, 1]) * self.rew_scales["roll"]  
        rew_action_rate = torch.sum(torch.square(self.last_actions[index, :] - self.actions[index, :]), dim=1) * self.rew_scales["action_rate"]

        # total reward
        self.rew_buf[index] += rew_joint_acc + rew_torque + rew_fallen_over + rew_height
  
        # log episode reward sums
        self.episode_sums["torques"][index] += rew_torque
        self.episode_sums["joint_acc"][index] += rew_joint_acc
        self.episode_sums["height"][index] += rew_height
        self.episode_sums["action_rate"][index] += rew_action_rate
        self.episode_sums["roll"][index] += rew_roll

        # print("ZERO")


    def first_landing_fun(self):    
        index = self.condition_invert # self.condition_first_landing   
        self.first_landing[index] = 1
        self.first_landing_time[index] = self.progress_buf[index]


    def calculate_metrics_first(self):
        
        index = self.condition_ground

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel[index, :] - self.dof_vel[index, :]), dim=1) * self.rew_scales["joint_acc"] 

        # torque penalty  
        rew_torque = torch.sum(torch.square(self.torques[index, :]), dim=1) * self.rew_scales["torque"]  

        # fallen over penalty
        rew_fallen_over = self.has_fallen[index] * self.rew_scales["fallen_over"]

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions[index, :] - self.actions[index, :]), dim=1) * self.rew_scales["action_rate"]
       
        # roll penalty           
        rew_roll = torch.square(self.base_ang_vel[index, 1]) * self.rew_scales["roll"]  

        # hasn't flew penalty
        rew_hasnt_flew = self.hasnt_flew[index] * self.rew_scales["hasnt_flew"]

        # velocity error in order to perfor the jump
        rew_velocity_x = 1 / (1 + torch.square(self.base_lin_vel[index, 0] - self.base_lin_target[index, 0])) * self.rew_scales["velocity_x"]
        rew_velocity_z = 1 / (1 + torch.square(self.base_lin_vel[index, 2] - self.base_lin_target[index, 2])) * self.rew_scales["velocity_z"] 

        # total reward
        self.rew_buf[index] += rew_joint_acc + rew_torque + rew_fallen_over + rew_action_rate + rew_roll + rew_hasnt_flew + rew_velocity_x + rew_velocity_z

  
        # log episode reward sums
        self.episode_sums["torques"][index] += rew_torque
        self.episode_sums["joint_acc"][index] += rew_joint_acc
        self.episode_sums["action_rate"][index] += rew_action_rate
        self.episode_sums["roll"][index] += rew_roll
        self.episode_sums["velocity_z"][index] += rew_velocity_z
        self.episode_sums["velocity_x"][index] += rew_velocity_x
        
        # print("UNO")

    def first_takeoff_fun(self):
        index = self.condition_first_takeoff
        self.first_takeoff[index] = 1
        self.jump_time[index] = self.progress_buf[index]
        self.has_early_landed[index] = 0

    def early_landing(self):
        index = self.condition_early_landing
        self.has_early_landed[index] = 1
        self.first_takeoff[index] = 0

 
    def calculate_metrics_second(self):

        index = self.condition_fly
       
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel[index, :] - self.dof_vel[index, :]), dim=1) * self.rew_scales["joint_acc"] 
        #following defined trajectory
        time = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        time[index] = self.progress_buf[index] - self.jump_time[index]

        # start_time_condition = (self.progress_buf - self.jump_time == 0)

        # if self.count%192 == 0 & start_time_condition.any():
        #     print("target_velZ", (self.base_lin_target[start_time_condition, 2]).mean())
        #     print("target_velX", (self.base_lin_target[start_time_condition, 0]).mean())
        #     print("diff_vel_z", (self.base_lin_vel[start_time_condition, 2] - self.base_lin_target[start_time_condition, 2]).mean())  
        #     print("diff_vel_x", (self.base_lin_vel[start_time_condition, 0] - self.base_lin_target[start_time_condition, 0]).mean())             

        true_time = time * self.dt

        # print("time", true_time)
        
        time_condition = (true_time > 0.1) & (self.has_early_landed == 0)

        if time_condition.any():

            self.computed_dist = torch.zeros(self._num_envs, device=self._device)
            self.computed_dist = self.dist_to_traj(self.base_pos[:, 0:3], self.trajectory, time)
            rew_trajectory = torch.zeros(len(index), device=self._device)
            rew_trajectory += -0.03
            rew_trajectory_all = (self.computed_dist * self.rew_scales["trajectory"]).reshape(self.num_envs) 

            rew_trajectory[time_condition] += rew_trajectory_all[time_condition].to(self.device)

        else:
            rew_trajectory = torch.zeros(len(index), device=self._device)
            rew_trajectory += -0.03
            

        rew_torque = torch.sum(torch.square(self.torques[index, :]), dim=1) * self.rew_scales["torque"] 
        rew_roll = torch.square(self.base_ang_vel[index, 1]) * self.rew_scales["roll"] 
        rew_time = torch.log10(1 + time[index]) * self.rew_scales["time"] 
        rew_fallen_over = self.has_fallen[index] * self.rew_scales["fallen_over_final"]

        # if self.count%192 == 0 :
        #     print("dist_to_traj", self.computed_dist[index].mean().numpy())
        #     print("true_time", (sum(true_time[index])/len(true_time[index])).numpy())  

        # total reward
        self.rew_buf[index]  += rew_joint_acc + rew_torque + rew_roll + rew_time + rew_trajectory[index] + rew_fallen_over

        # log episode reward sums

        self.episode_sums["torques"][index] += rew_torque
        self.episode_sums["joint_acc"][index] += rew_joint_acc
        self.episode_sums["roll"][index] += rew_roll
        self.episode_sums["trajectory"][index] += rew_trajectory[index]
        self.episode_sums["time"][index] += rew_time

        # print("DUE")

    def calculate_metrics_third(self):
    
        index = self.condition_second_landing

        rew_reached_last_phase = self.condition_second_landing[index] * 2.5

        # mission done
        rew_mission_complete = self.mission_complete[index] * self.rew_scales["termination"]

        # set velocity to zero
        rew_velocity = torch.sum(torch.square(self.base_lin_vel[index, :]), dim=1) * self.rew_scales["velocity_final"]

        rew_fallen_over = self.has_fallen[index] * self.rew_scales["fallen_over_final"]

        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel[index, :]  - self.dof_vel[index, :] ), dim=1) * self.rew_scales["joint_acc"]         
            
        rew_torque = torch.sum(torch.square(self.torques[index, :] ), dim=1) * self.rew_scales["torque"] 

        # ground forces over feet
        rew_ground_force = torch.sum(torch.norm(torch.reshape( self._solos.get_contact_forces()[0], (1024, 4, 3))[index] , dim=2), dim=1) * self.rew_scales["ground_forces"] 

        # total reward
        self.rew_buf[index] += rew_joint_acc + rew_torque + rew_mission_complete + rew_ground_force + rew_velocity + rew_fallen_over + rew_reached_last_phase
        # log episode reward sums

        self.episode_sums["torques"][index] += rew_torque
        self.episode_sums["joint_acc"][index] += rew_joint_acc
        self.episode_sums["velocity_final"][index] += rew_velocity
        self.episode_sums["ground_force"][index] += rew_ground_force


    def dist_to_traj(self, point, traj, time_buf):

        traj_tensor = traj.to(point.device) 
        instant = (time_buf * self.dt / self.delta_t_traj_s).long()
        dists = torch.zeros(self._num_envs)
        instant = torch.clamp(instant, 0, traj_tensor.size()[2] - 1)
        selected_traj = torch.zeros(self._num_envs, 3, device = self._device)
        
        for i in range(len(instant)):
            selected_traj = traj_tensor[:, :, instant[i]]

        dists = torch.norm(selected_traj - point, dim=1) 
        return dists 

    def create_trajectory(self, commands, env_ids):

        finish = torch.zeros(self._num_envs, 3, device = self._device)
        start = (self.base_pos[:, 0:3]).to(finish.device)
        start[:, 2] = self.robot_default_height 

        # make start a bit further
        # start[:, 0] += 0.3  
        # start[:, 2] += 0.1

        finish[:, 2] = commands + self.robot_default_height
        finish[:, 1] = start[:, 1]
        finish[:, 0] = start[:, 0] - self.random_position.t()
 
        # Calculate the distance between start and finish points
        distance_norm = torch.norm(finish - start)

        # set maximum height
        maximum_height = (commands + self.robot_default_height) * (1 + commands/distance_norm) # calculate it given distance, need to think

        num_points = 10
        t = torch.linspace(0, 1, num_points).to(self._device)

        vel_zeta = (torch.sqrt(torch.abs((maximum_height - start[:, 2]) * (2*self.gravity)))).to(self._device)

        final_time = (- vel_zeta - torch.sqrt(torch.abs(torch.pow(vel_zeta, 2) + 2 * self.gravity + (finish[:, 2] - start[:, 2])))) / self.gravity

        time = final_time.unsqueeze(1) * t.unsqueeze(0)

        self.delta_t_traj_s = final_time / num_points

        start_value = start[0, 2]
        vel_zeta_value = vel_zeta[0]
        time_values = time[0]

        zeta_axis_intermediate = start_value + vel_zeta_value * time_values
        gravity_term = 0.5 * self.gravity * torch.pow(time_values, 2)
        zeta_axis = (zeta_axis_intermediate + gravity_term)

        vel_x = ((finish[:, 0] - start[:, 0]) / final_time).to(self._device)

        vel_y = ((finish[:, 1] - start[:, 1]) / final_time).to(self._device)

        x_axis = (start[:, 0] + vel_x.unsqueeze(0) * time_values.unsqueeze(1))

        y_axis = torch.tile(start[:, 1].unsqueeze(1), (1, num_points))

        self.base_lin_target = torch.stack([vel_x.clone().detach(), vel_y.clone().detach(), vel_zeta.clone().detach()], dim=1)
  
        self.trajectory = torch.stack ([x_axis.t(), y_axis, zeta_axis.unsqueeze(0).expand(self._num_envs, -1)], dim = 1)


    def get_observations(self):
        self.measured_heights = self.get_heights()
        heights = torch.clip(self.base_pos[:, 2].unsqueeze(1) - self.robot_default_height - self.measured_heights, -1, 1.) * self.height_meas_scale
        self.obs_buf = torch.cat((  self.base_lin_vel * self.lin_vel_scale,
                                    self.base_ang_vel  * self.ang_vel_scale,
                                    self.projected_gravity,
                                    self.commands[:, 0:2], # * self.commands_scale,
                                    self.dof_pos * self.dof_pos_scale,
                                    self.dof_vel * self.dof_vel_scale,
                                    heights,
                                    self.actions
                                    ),dim=-1)
    
    def get_ground_heights_below_knees(self):
        points = self.knee_pos.reshape(self.num_envs, 4, 3)
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
    
    def get_ground_heights_below_base(self):
        points = self.base_pos.reshape(self.num_envs, 1, 3)
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
                                    
    def get_heights(self, env_ids=None):
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.base_pos[:, 0:3]).unsqueeze(1)
 
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))
