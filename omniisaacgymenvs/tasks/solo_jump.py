from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.anymal import Anymal
from omniisaacgymenvs.robots.articulations.views.anymal_view import AnymalView
from omniisaacgymenvs.tasks.utils.terrain_jump import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.simulation_context import SimulationContext

import numpy as np
import torch
import math


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
    
        self.rew_scales["roll"] = self._task_cfg["env"]["learn"]["rollRewardScale"]
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["TerminationRewardScale"]


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
        self.base_threshold = 0.2
        self.knee_threshold = 0.1
        self.max_ground_time_s = 2 #sec
        self.max_ground_time = int(self.max_ground_time_s/ self.dt + 0.5)
        self.flying_threshold = 1 # meters
        
   

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_observations = 189
        self._num_actions = 12

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
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # distance, yaw, height, type terrain
        # self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.gravity_vec = torch.tensor(get_axis_params(-1., self.up_axis_idx), dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.last_base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)

        self.num_dof = 20  # 20(flatfoot)    24(3dof_flatfoot)
        self.all_torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        self.height_points = self.init_height_points()
        self.measured_heights = None
        # joint positions offsets
        self.default_dof_pos = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)
        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = { "torques": torch_zeros(), "joint_acc": torch_zeros(), "action_rate": torch_zeros(), "roll": torch_zeros(), "trajectory": torch_zeros() }
        return


    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:13] = 0. # commands
        noise_vec[13:25] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[25:37] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[37:177] = self._task_cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        noise_vec[177:189] = 0. # previous actions
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
        self.get_anymal()
        super().set_up_scene(scene)
        self._anymals = AnymalView(prim_paths_expr="/World/envs/.*/anymal", name="anymal_view", track_contact_forces=True)
        scene.add(self._anymals)
        scene.add(self._anymals._knees)
        scene.add(self._anymals._base)

    def get_terrain(self):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if not self.curriculum: self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1

        self.terrain_levels = torch.randint(0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)

        self.commands[:, 2] = 1.5 + self.terrain_levels * 0.3
        self.commands[:, 3] = self.terrain_types

        self._create_trimesh()
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            
    def get_anymal(self):
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False)
        # aggiunte mie
        anymal_translation = torch.tensor([0.0, 0.0, 0.0])  # self.anymal_init_pos
        anymal_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])  # self.anymal_init_orientation
        anymal = Anymal(prim_path=self.default_zero_env_path + "/anymal", 
                        name="anymal",
                        translation=anymal_translation, 
                        orientation=anymal_orientation,
                        softfoot=True)
        self._sim_config.apply_articulation_settings("anymal", get_prim_at_path(anymal.prim_path), self._sim_config.parse_actor_config("anymal"))
        anymal.set_anymal_properties(self._stage, anymal.prim)
        anymal.prepare_contacts(self._stage, anymal.prim)

        self.dof_names = anymal.dof_names
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def post_reset(self):
        for i in range(self.num_envs):
            self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
        # self.num_dof = self._anymals.num_dof
        self.dof_pos = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device)

        self.all_dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.all_dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)

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

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), 12), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), 12), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        self.update_terrain_level(env_ids)
        self.base_pos[env_ids] = self.base_init_state[0:3]
        self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
        self.base_pos[env_ids, 0:1] += torch_rand_float(-2, -1, (len(env_ids), 1), device=self.device)

        
        self.base_quat[env_ids] = self.base_init_state[3:7]
        angle = torch_rand_float(0, 2*3.14, (len(env_ids), 1), device=self.device) / 2
        self.base_quat[env_ids, 0:1] = torch.cos(angle)
        self.base_quat[env_ids, 3:4] = torch.sin(angle)

        self.commands[env_ids, 0:1] = self.base_pos[env_ids, 0:1] - self.env_origins[env_ids, 0:1]
        # self.commands[env_ids, 0:2] = 0
        self.commands[env_ids, 1:2] = 2 * torch.atan2(self.base_quat[env_ids, 3:4], self.base_quat[env_ids, 0:1]) # malloc problem BOH

        self.trajectory = self.create_trajectory(self.commands[:, 2])

        self.base_velocities[env_ids] = self.base_init_state[7:]

        self._anymals.set_world_poses(positions=self.base_pos[env_ids].clone(), 
                                      orientations=self.base_quat[env_ids].clone(),
                                      indices=indices)
        self._anymals.set_velocities(velocities=self.base_velocities[env_ids].clone(),
                                          indices=indices)
        self.all_dof_pos[env_ids, :12] = self.dof_pos[env_ids]
        self._anymals.set_joint_positions(positions=self.all_dof_pos[env_ids].clone(), 
                                          indices=indices)
        self.all_dof_vel[env_ids, :12] = self.dof_vel[env_ids]
        self._anymals.set_joint_velocities(velocities=self.all_dof_vel[env_ids].clone(), 
                                          indices=indices)

        # self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
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
        root_pos, _ = self._anymals.get_world_poses(clone=False)
        distance = torch.norm(root_pos[env_ids, 0:3] - self.env_origins[env_ids, 0:3], dim=1)

        # self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
        # self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)

        self.terrain_levels[env_ids] -= 1 * (distance > 2)
        self.terrain_levels[env_ids] += 1 * (distance < 0.4)

        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def refresh_dof_state_tensors(self):
        self.dof_pos = self._anymals.get_joint_positions(clone=False)[:, :12]
        self.dof_vel = self._anymals.get_joint_velocities(clone=False)[:, :12]
    
    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._anymals.get_world_poses(clone=False)
        self.base_velocities = self._anymals.get_velocities(clone=False)
        self.knee_pos, self.knee_quat = self._anymals._knees.get_world_poses(clone=False)

    def pre_physics_step(self, actions):
        if not self._env._world.is_playing():
            return

        self.actions = actions.clone().to(self.device)
        for i in range(self.decimation):
            if self._env._world.is_playing():
                torques = torch.clip(self.Kp*(self.action_scale*self.actions + self.default_dof_pos - self.dof_pos) - self.Kd*self.dof_vel, -80., 80.)
                self.all_torques[:, :12] = torques
                self._anymals.set_joint_efforts(self.all_torques)
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
            self.calculate_metrics()

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()
            if self.add_noise:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

            self.last_actions[:] = self.actions[:]
            self.last_dof_vel[:] = self.dof_vel[:]


        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def push_robots(self):
        self.base_velocities[:, 0:2] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device) # lin vel x/y
        self._anymals.set_velocities(self.base_velocities)
    
    def check_termination(self):
        #self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))
        knee_contact = torch.norm(self._anymals._knees.get_net_contact_forces(clone=False).view(self._num_envs, 4, 3), dim=-1) > 1.
        self.has_fallen = (torch.norm(self._anymals._base.get_net_contact_forces(clone=False), dim=1) > 1.) | (torch.sum(knee_contact, dim=-1) > 1.)
        self.reset_buf = self.has_fallen.clone().int()
        #self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

        self.hasnt_flew = (self.progress_buf >= self.max_ground_time).to(torch.int) & self._anymals.is_base_below_threshold(self.flying_threshold, 0.0).to(torch.int)
        self.reset_buf += self.hasnt_flew.clone().int()
  

        self.mission_complete = ((self.base_pos[:, 2:3] - self.commands[:, 2:3] < 0.1).t() & (self.base_lin_vel < 0.1).all(dim=1)).flatten() # if vel low and heigth position reached
        self.reset_buf += self.mission_complete.clone()
        # self.reset_buf = torch.ones_like(self.reset_buf)

    def calculate_metrics(self):

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"] 

        # distance from trajectory penalty
        self.computed_dist = self.dist_to_traj(self.base_pos[:, 0:3], self.trajectory)
        base_pos_tensor = self.base_pos[:, 2:3].clone().detach()
        base_pos_tensor_T = torch.transpose(base_pos_tensor, 0, 1)  
        base_pos_tensor_T = base_pos_tensor_T.to(self.computed_dist.device)
        rew_trajectory = (self.computed_dist * self.rew_scales["trajectory"] * base_pos_tensor_T).reshape(self.num_envs) 
        rew_trajectory = rew_trajectory.to(self.device)

        # torque penalty  
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]  

        # fallen over penalty
        rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
       
        # roll penalty           
        rew_roll = torch.square(self.base_ang_vel[:, 1]) * self.rew_scales["roll"]  

        # right final position reward
        rew_mission_complete = self.mission_complete * self.rew_scales["termination"]

        # hasn't flew penalty
        rew_hasnt_flew = self.hasnt_flew * self.rew_scales["hasnt_flew"]

        # total reward
        self.rew_buf = rew_joint_acc + rew_torque + rew_fallen_over + rew_action_rate + rew_roll + rew_hasnt_flew + rew_mission_complete + rew_trajectory
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # log episode reward sums

        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["roll"] += rew_roll
        self.episode_sums["trajectory"] += rew_trajectory

 
    def dist_to_traj(self, point, traj):
        traj_tensor = torch.from_numpy(traj).to(point.device)  # Convert traj to torch tensor and move to the same device as point
        dists = torch.norm(traj_tensor - point.unsqueeze(2), dim=1)  # Compute L2 distance between each point in traj and point, resulting in a tensor of shape (num_envs, num_points)
        return torch.min(dists, dim=1)[0]  # Compute the minimum distance along the second dimension, resulting in a tensor of shape (num_envs,)

    def create_trajectory(self, commands):
        finish = torch.zeros(self.num_envs, 3)
        start = (self.base_pos[:, 0:3]).to(finish.device)
        start[:, 2] += 0.5
        finish[:, 2] = commands
        # Calculate the distance between start and finish points
        distance_norm = torch.norm(finish - start)
        distance = finish - start
        # Calculate the direction of the trajectory
        direction = distance / torch.sqrt(distance_norm)
        # set maximum height
        maximum_height = commands * (1 + commands/distance_norm) # calculate it given distance, need to think

        num_points = 20
        t = np.linspace(0, 1, num_points)
        x = start[:, 0].reshape(-1, 1) + direction[:, 0].reshape(-1, 1) * distance[:, 0].reshape(-1, 1) @ t.reshape(1, -1)
        y = start[:, 1].reshape(-1, 1) + direction[:, 1].reshape(-1, 1) * distance[:, 1].reshape(-1, 1) @ t.reshape(1, -1)
        plane = np.vstack((x, y)).transpose().reshape(self.num_envs, num_points, 2)
  
        a = (finish[:, 2] - start[:, 2]) / np.linalg.norm(finish[:, 0:2] - start[:, 0:2])  
        b = (start[:, 2] - a * np.linalg.norm(start[:, 0:2]) + a * np.linalg.norm(finish[:, 0:2] - start[:, 0:2])).to(a.device) 
        c = (maximum_height.to(a.device) - b)
   
        plane = torch.from_numpy(plane).to(a.device)
 
        z = a[:, None] * torch.norm(plane, dim=2) + b[:, None] + c[:, None]

        trajectory = np.stack((x, y, z), axis=1)  # num_env x 3D x t

        return trajectory




    def get_observations(self):
        self.measured_heights = self.get_heights()
        heights = torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.height_meas_scale
        self.obs_buf = torch.cat((  self.base_lin_vel * self.lin_vel_scale,
                                    self.base_ang_vel  * self.ang_vel_scale,
                                    self.projected_gravity,
                                    self.commands[:, 0:4], # * self.commands_scale,
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
