from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.softleg import Softleg
from omniisaacgymenvs.robots.articulations.views.softleg_view import JumpingSoftlegView
# from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.simulation_context import SimulationContext
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
import torch
import csv
import os
import math
from pxr import PhysxSchema

### THIS HAS BEEN TESTED IN THE REAL EXPERIMENTS

class SoftlegJumpTaskCinque(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        n_joints=3,
        n_actuators=2,
        WANNA_PRINT_WAY_2_MUCH=False,
        PLT_ONE=True,
        USE_DRIVERS=True,
        WANNA_SPEED_UP=True,
        WANNA_INFO = False,
        offset=None, 
    ) -> None:
               
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self.USE_DRIVERS = USE_DRIVERS
        self.PRINT_OBS_FUN = True 
        self.WANNA_INFO = WANNA_INFO
        self.INIT_REW_PARTIAL = True 
        self.WANNA_PRINT_WAY_2_MUCH = WANNA_PRINT_WAY_2_MUCH
        self.PLT_ONE = PLT_ONE
        self.WANNA_SPEED_UP = WANNA_SPEED_UP
        self._task_cfg = sim_config.task_config
        self._n_actuators = n_actuators
        self._n_joints = n_joints
        self._n_state = 2 * self._n_joints
        self._device = 'cuda:0'
        self._count = 0
        self.l2 = 0.19
        # SoftlegJump040_target12
        self._num_observations = 7 # [q1 q2, q1_vel q2_vel, actions, time] 
        self._num_actions = self._n_actuators  # two actuators
        self._offset_landing = 2.0 
        self._cart_joint_friction = 0.2 # 0.4
        
        # print(self._cfg['name'])
        # print(self._cfg['task']) # this prints all 
        # print(self._cfg['checkpoint'])
        if self._cfg["test"]:
            self.filename_2save = os.path.splitext(os.path.basename(self._cfg['checkpoint']))[0]          
      
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"] 
        self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"]["staticFriction"]
        self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"]["dynamicFriction"]
        self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"]["restitution"]
        self._task_cfg["sim"]["add_ground_plane"] = True
        self.gravity = torch.tensor(self._task_cfg["sim"]["gravity"][2], device=self._device)
        self._height_des = self._task_cfg["env"]["heightDes"]
        
        ## normalization terms in my state 
        self._q_scale = self._task_cfg["env"]["learn"]["qScale"]
        self._q_dot_scale = self._task_cfg["env"]["learn"]["qDotScale"]
        self._action_scale = self._task_cfg["env"]["control"]["actionScale"]
        self._length_strait_softleg = self._task_cfg["env"]["lengthSoftLeg"] # [m] 0.255 + 0.17 + 0.01, thigh shank foot
        self._flying_threshold = self._length_strait_softleg 
        self._when_to_print = self._task_cfg["env"]["whenToPrint"] 
        
        ## reward scale 
        self.rew_scales = {}
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["height"] = self._task_cfg["env"]["learn"]["storageheightRewardScale"]   
        self.rew_scales["velocity_final"] = self._task_cfg["env"]["learn"]["velocityRewardScale"]   
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminationRewardScale"]
        self.rew_scales["time"] = self._task_cfg["env"]["learn"]["timeRewardScale"]
        self.rew_scales["ground_force"] = self._task_cfg["env"]["learn"]["ground_forcesRewardScale"]
        self.rew_scales["final_config"] = self._task_cfg["env"]["learn"]["finalConfig"]
        self.rew_scales["foot_down_prismatic"] = self._task_cfg["env"]["learn"]["footDownPrismatic"]
        self.rew_scales["hip_minus_all_time"] = self._task_cfg["env"]["learn"]["hipMinusAllTheTime"]
        self.rew_scales["distance"] = self._task_cfg["env"]["learn"]["distanceScale"]

        ## initial config 
        self.q_init = self._task_cfg["env"]["initState"]["q"]
        self.q_dot_init = self._task_cfg["env"]["initState"]["qDot"]
        self._default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]
        self._default_joint_angles_double = self._task_cfg["env"]["initState"]["q"]
        self._init_state = self.q_init + self.q_dot_init       
        self._reset_dist = self._task_cfg["env"]["resetDist"] 
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_obs = self._task_cfg["env"]["clipObservations"] - 20
        self._decimation = self._task_cfg["env"]["control"]["decimation"]
        self._sim_dt = self._task_cfg["sim"]["dt"]
        self._dt = self._decimation * self._sim_dt
        self._pretime_sec = self._task_cfg["sim"]["preTime"]
        self._max_episode_length_s = self._task_cfg["env"]["episodeLength"] 

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self._dt
            
        ## low level control gains 
        self._Kp = self._task_cfg["env"]["control"]["stiffness"]
        self._Kd = self._task_cfg["env"]["control"]["damping"]
        
        self.extras = {}
        
        self._w_control_var_sat = self._task_cfg["env"]["learn"]["wControlVarSat"]
        self._w_control_var = self._task_cfg["env"]["learn"]["wControlVar"]
        self._w_error_reaching = self._task_cfg["env"]["learn"]["wErrDistance"]
        self._w_config_start_end = self._task_cfg["env"]["learn"]["wConfigStartEnd"]
        self._w_vel_final = self._task_cfg["env"]["learn"]["wVelFinal"]
        self._w_height_general = self._task_cfg["env"]["learn"]["wHeightGneral"]
        self._w_foot_down_prismatic = self._task_cfg["env"]["learn"]["wFootDownPrismatic"]
        self._w_hip_minus_all_time = self._task_cfg["env"]["learn"]["wHipMinusAllTime"]
        self._w_ground_force = self._task_cfg["env"]["learn"]["wGroundForce"]
        self._w_action_rate = self._task_cfg["env"]["learn"]["wActionRate"]
        self._w_joint_acc = self._task_cfg["env"]["learn"]["wJointAcc"]

        self.timeout_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._first_landing = torch.zeros(self._num_envs, dtype=torch.int, device=self._device)
        self._not_evaluated = torch.zeros(self._num_envs, dtype=torch.int, device=self._device)
        self._has_early_landed = torch.zeros(self._num_envs, dtype=torch.int, device=self._device)
        self._jump_time = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._foot_air_time = torch.zeros(self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._last_config = torch.zeros(self._num_envs, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        self._last_config_des = torch.zeros(self._num_envs, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        self._high_softlegs = torch.zeros(self._num_envs, int(self._max_episode_length_s/self._dt) + 2, dtype=torch.float, device=self._device, requires_grad=False)
        self._q_init_env = torch.zeros(self._num_envs, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        self.epoch_num = 0
        self._cond_jump_and_reach = torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device)
        self._has_landed = torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device)
        self.check_one_jump = torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device)
        
        RLTask.__init__(self, name, env)  

        self._noise_scale_vec = self.get_noise_scale_vec(self._task_cfg)
        
        self._torques = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self.actions = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._last_actions = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._last_joint_vel = torch.zeros(self._num_envs, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        torch_zeros = lambda : torch.zeros(self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)
        self._default_dof_tensors = torch.stack([torch.tensor(self.q_init, dtype=torch.long, device=self._device)] * self._num_envs, dim=0)
        
        ## to do insiema agli extras
        self.episode_sums = {"distance": torch_zeros(), "height": torch_zeros(), "time": torch_zeros(), "ground_force": torch_zeros(), "torque": torch_zeros(), 
                            "action_rate": torch_zeros(), "velocity_final":torch_zeros(), "hip_minus_all_time": torch_zeros(),
                            "joint_acc": torch_zeros(), "joint_vel": torch_zeros(), "joint_pos": torch_zeros(), 
                            "max_reached_height" : torch_zeros(), "final_config": torch_zeros(), "foot_down_prismatic": torch_zeros()}
        
        return
    
    def get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self._add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        
        noise_vec[0] = self._task_cfg["env"]["learn"]["qNoise"] * noise_level / 10
        noise_vec[1 : self._n_joints] = self._task_cfg["env"]["learn"]["qNoise"] * noise_level * self._q_scale
        
        noise_vec[self._n_joints] = self._task_cfg["env"]["learn"]["qDotNoise"] * noise_level / 10
        noise_vec[self._n_joints + 1 : 2 * self._n_joints] = self._task_cfg["env"]["learn"]["qDotNoise"] * noise_level * self._q_dot_scale / 10
        noise_vec[2 * self._n_joints + 1 : 2 * self._n_joints + self._num_actions] = self._task_cfg["env"]["learn"]["qNoise"] * noise_level / 10
        return noise_vec
    
    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage() 
        self.get_softleg()
        super().set_up_scene(scene)
        self._softlegs = JumpingSoftlegView(prim_paths_expr="/World/envs/.*/softleg_cart", 
                                            name="JumpingSoftlegView", 
                                            default_dof_pos = self._default_joint_angles_double,
                                            track_contact_forces=True) 
        scene.add(self._softlegs)
        scene.add(self._softlegs._foot)
        scene.add(self._softlegs._cart)
                
        for i in range(0, self._num_envs):
            joint_api = PhysxSchema.PhysxJointAPI.Get(self._stage, '/World/envs/env_' + str(i) + '/softleg_cart/softleg_1_cart_link/softleg_1_cart_joint')
            if not joint_api.GetJointFrictionAttr():
                joint_api.CreateJointFrictionAttr(self._cart_joint_friction)
            else:
                joint_api.GetJointFrictionAttr().Set(self._cart_joint_friction)
        
    def get_softleg(self):
        self._init_state = torch.tensor(self._init_state, dtype=torch.float, device=self._device, requires_grad=False)
        softleg = Softleg(prim_path=self.default_zero_env_path + "/softleg_cart", name="Softleg")
        if self.WANNA_PRINT_WAY_2_MUCH:
            print(dir(softleg))
        self._sim_config.apply_articulation_settings("Softleg",
                                                     get_prim_at_path(softleg.prim_path),
                                                     self._sim_config.parse_actor_config("Softleg"))
        
        softleg.set_softleg_properties(self._stage, softleg.prim)
        softleg.prepare_contacts(self._stage, softleg.prim)
        self.dof_names = softleg.dof_names
                
        self._name_joint = ['softleg_1_cart_joint', 
                            'softleg_1_hip_joint', 
                            'softleg_1_knee_joint']
        
        self._dof_names_properties = ["softleg_1_cart_link/softleg_1_cart_joint",
                                    "softleg_1_base_link/softleg_1_hip_joint",
                                    "softleg_1_thigh_link/softleg_1_knee_joint"]
                    
        drive_type = ["linear"] + ["angular"] * 2
        
        # stiffness = [0.0] + [1.23] * 2
        # damping = [0.0] + [0.006981317] * 2
        ## from SoftlegJump030_target9
        
        stiffness = [0.0] + [0.8674531575294934] * 2
        damping = [0.0] + [0.08674531575294935] * 2
        max_force = [0.0] + [20.0] * 2
                
        for i, dof in enumerate(self._dof_names_properties):
            set_drive(
                prim_path=f"{softleg.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=self._default_joint_angles_double[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )
                
        for i in range(self._n_joints):
            name = self.dof_names[i]
            config_softleg = self._default_joint_angles[name]
            self._default_dof_tensors[:, i] = config_softleg
            
    @staticmethod
    def compute_height(q1, q2, l1=0.175, l2=0.19):
        return l1 * torch.sin(q1) + l2 * torch.sin(q1 + q2)
    
    def refresh_dof_state_tensors(self):
        self.dof_pos = self._softlegs.get_joint_positions(clone=False) + self.set_offset_prismatic_config()
        self.dof_vel = self._softlegs.get_joint_velocities(clone=False)
        self.contact_force_foot = self._softlegs._foot.get_net_contact_forces(dt=self._dt, clone=False)
        pos_foot_only, _ = self._softlegs._foot.get_world_poses(clone=False) # this is the knee actually
        self.from_knee_to_foot = pos_foot_only[:,-1]
        
        if self._cfg['test'] & self.WANNA_INFO:
            ## debug
            print('Foot Pose: \n', self.from_knee_to_foot, '\n')
            print('Base Pose: \n', self.dof_pos[:,0], '\n')
            print('Cantact Z: \n', self.contact_force_foot[:,-1].flatten(), '\n')
            print('=======================================================================================================')
        
    def get_foot_position(self):
        self.knee_pos, _ = self._softlegs._foot.get_local_poses() ## this is very time-consuming ## get_world_pose
        from_knee_to_foot = self.l2 * torch.sin(self.dof_pos[:,-1])
        component_x = torch.zeros(self._num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        from_knee_to_foot_3D = torch.cat((component_x.unsqueeze(1), component_x.unsqueeze(1), from_knee_to_foot.unsqueeze(1)), dim=1)
        self.foot_pose = self.knee_pos + from_knee_to_foot_3D
        
    def get_observations(self) -> dict:            
        self.refresh_dof_state_tensors()
        if self.WANNA_SPEED_UP:
            pass
        else:
            self.get_foot_position()
            
        self.obs_buf[:, 0 : self._n_joints - 1] = self.dof_pos[:, 1 : self._n_joints] / self._q_scale
        self.obs_buf[:, self._n_joints - 1 : 2 * self._n_joints - 2] = self.dof_vel[:, 1 : self._n_joints] / self._q_dot_scale ## max velocity allowed
        self.obs_buf[:, 2 * self._n_joints - 2: 2 * self._n_joints + self._num_actions - 2] = self._last_actions
        self.obs_buf[:, -1] = (self._max_episode_length_s - self.progress_buf * self._dt) / self._max_episode_length_s
        # input(self.obs_buf)
        observations = {self._softlegs.name: {"obs_buf": self.obs_buf}}
        
        return observations 
    
    def set_offset_prismatic_foot(self, _off_set = 2.0):
        return torch.tensor([[0.0, 0.0, _off_set]] * self._num_envs, device=self._device)
    
    def set_offset_prismatic_config(self, _off_set = 2.0):
        return torch.tensor([[_off_set, 0.0, 0.0]] * self._num_envs, device=self._device)
    
    def pre_physics_step(self, actions) -> None:
        ## Here the agent starts to learn 
     
        if not self._env._world.is_playing():
            return
        
        self.actions = actions.to(self._device)  
        # if self._cfg['test']:
        #     input(self.actions)
               
        for _ in range(self._decimation):
            if self._env._world.is_playing():
                if self.USE_DRIVERS:
                    ## with drivers NOTE THAT THIS IS A POSITION NOT A TORQUE
                    torques = self._action_scale * self.actions
                    # from SoftlegJump040_target12
                    self._softlegs.set_joint_position_targets(torques, joint_indices=[self._softleg_hip_idx, self._softleg_knee_idx])
                    # self._softlegs.set_joint_positions(torques, joint_indices=[self._softleg_hip_idx, self._softleg_knee_idx])
                    self._torques = torques
                else:
                    ## without drivers, this has never not been used we are setting a torque
                    torques = torch.clip(self._Kp * (self._action_scale * self.actions + self._default_dof_tensors[:, 1:] - self.dof_pos[:, 1:]) - self._Kd * self.dof_vel[:, 1:], -20., 20.)
                    zeros_column = torch.zeros(torques.shape[0], 1, device=torques.device)
                    torques = torch.cat((zeros_column, torques), dim=1)
                    self._torques = torques
                    self._softlegs.set_joint_efforts(torques)
                
                SimulationContext.step(self._env._world, render=False)
                self.refresh_dof_state_tensors()
                if self.WANNA_SPEED_UP:
                    pass
                else:
                    self.get_foot_position()
                
                if self._num_envs < 1:
                    if self._count%self._when_to_print == 0: ## this _count has no physical meaning use self.progress_buf instead 
                        print('=======================================================================================================')
                        print('Config Robot:\n\t{}'.format(self.dof_pos))
                        print('Contac Force Foot:\n\t{}'.format(self.contact_force_foot))
                        print('Torques:\n\t{}'.format(torques))
                        print('Counter: ', self._count, '\tdt: ', self._dt, '\tTime: [s]', self._count * self._dt)
                        print('=======================================================================================================')
                # self._count += 1
    
    
    def compute_q2(self, num_resets, q1, offset=torch.pi/20):
        lower_bound = -(torch.pi + q1 - offset)
        upper_bound = -q1 - offset
        random_tensor = torch.rand(num_resets, device=q1.device)  # Use the device of q1
        # random_tensor = random_tensor * (upper_bound - lower_bound) + lower_bound
        # q2 = - math.pi/2 * torch.rand(num_resets, device=self._device)
        q2 = random_tensor * (upper_bound - lower_bound) + lower_bound
        return q2
    
    
    def reset_idx(self, env_ids):
     
        num_resets = len(env_ids)
        self._count = 0
        self._high_softlegs = torch.zeros(num_resets, int(self._max_episode_length_s / self._dt) + 2, dtype=torch.float, device=self._device)
        # self._torque_softlegs = torch.zeros(num_resets, self._n_actuators, int(self._max_episode_length_s/self._dt) + 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._cond_jump_and_reach[env_ids] = False
        self._has_landed[env_ids] = False
        self.from_knee_to_foot = torch.zeros(num_resets, dtype=torch.float, device=self._device)
        self.check_one_jump[env_ids] = False
        
        # randomize DOF positions        
        self.dof_pos[env_ids, self._softleg_hip_idx] = -(math.pi) * torch.rand(num_resets, device=self._device) 
        self.dof_pos[env_ids, self._softleg_knee_idx] = self.compute_q2(num_resets, self.dof_pos[env_ids, self._softleg_hip_idx])
        self.dof_pos[env_ids, self._softleg_cart_idx] = torch.max(-self.compute_height(\
                                                        self.dof_pos[env_ids, self._softleg_hip_idx], self.dof_pos[env_ids, self._softleg_knee_idx]), \
                                                        torch.tensor(0.035, device=self._device)) + 0.002 - self._offset_landing 
            
        if self._cfg['test']:   
            self.dof_pos[env_ids, self._softleg_hip_idx] = -(math.pi) 
            self.dof_pos[env_ids, self._softleg_knee_idx] = 2.5
            self.dof_pos[env_ids, self._softleg_cart_idx] = 0.11 - 2.0
               
        self._last_config_des[env_ids, :] = self.dof_pos[env_ids, :] 
        self._q_init_env[env_ids, :] = self.dof_pos[env_ids, :] 
     
        self.dof_vel[env_ids, self._softleg_hip_idx] = 0 
        self.dof_vel[env_ids, self._softleg_knee_idx] = 0 
        self.dof_vel[env_ids, self._softleg_cart_idx] = 0 
        # self.actions[env_ids, :] = _offset_landing.dof_pos[env_ids, 1:] 
        
        self.dof_pos_save = self.dof_pos
        self.dof_vel_save = self.dof_vel
        self.actions = self.dof_pos[env_ids, 1:] / self._action_scale 
        self.actions_save = self._action_scale * self.actions
        self.contact_force_foot_save = torch.zeros(num_resets, 3, dtype=torch.float, device=self._device)

        indices = env_ids.to(dtype=torch.int32)
         
        self._softlegs.set_joint_positions(self.dof_pos, indices=indices)
        self._softlegs.set_joint_velocities(self.dof_vel, indices=indices)

        self._env._world.step(render=self._env._render)
        self.refresh_dof_state_tensors()

        # bookkeeping
        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0
        self._last_actions[env_ids,:] = self.dof_pos[env_ids, 1:] / self._action_scale # 0.     
        self._last_config[env_ids] = 0.
        self._last_joint_vel[env_ids] = 0.
                
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
            
    def post_reset(self):
       
        self._softleg_cart_idx = self._softlegs.get_dof_index("softleg_1_cart_joint")
        self._softleg_hip_idx = self._softlegs.get_dof_index("softleg_1_hip_joint")
        self._softleg_knee_idx = self._softlegs.get_dof_index("softleg_1_knee_joint")  
        
        self._num_dof = self._softlegs.num_dof
        self.dof_pos = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self.dof_vel = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self.actions = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self.from_knee_to_foot = torch.zeros(self._num_envs, dtype=torch.float, device=self._device)
        
        indices = torch.arange(self._softlegs.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def post_physics_step(self):
        self.progress_buf[:] += 1           
            
        if self._env._world.is_playing():
            self.refresh_dof_state_tensors()
            if self.WANNA_SPEED_UP:
                pass
            else:
                self.get_foot_position()
            self.is_done()
            self.get_states()
            self.calculate_metrics()
            
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)
                
            self.get_observations()
            if self.obs_buf.isnan().any():
                print(self.obs_buf.isnan().any(dim=0).nonzero(as_tuple=False).flatten())
            if self._add_noise:
                self.obs_buf += torch.rand_like(self.obs_buf) * self._noise_scale_vec

            self._last_actions[:] = self.actions[:]
            self._last_config[:] = self.dof_pos[:]   
            self._last_joint_vel[:] = self.dof_vel[:]        
        else:
            pass
        
        ## nvidia bugs
        # self._torque_softlegs[..., self._count] = self._softlegs.get_applied_joint_efforts(joint_indices=[self._softleg_hip_idx, self._softleg_knee_idx])
        self._count += 1
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
        
    
    def calculate_metrics(self) -> None:
        self._high_softlegs[:, self.progress_buf[0]] = self.dof_pos[:, 0]
        rew_action_rate = self.rew_scales["action_rate"] * torch.sum(torch.square(self.actions - self._last_actions), dim=-1) 
        rew_velocity = self._w_action_rate * self.rew_scales["action_rate"] * torch.sum(torch.square(self.dof_vel[:,1:-1]), dim=-1) 
        self.episode_sums["joint_vel"] += rew_velocity
        self.episode_sums["action_rate"] += rew_action_rate
        
        rew_ground = torch.where(((self.progress_buf * self._dt >= self._max_episode_length_s / 2) & (self.from_knee_to_foot < 0.015)), 0.5, 0.0)
        rew_vel_mio = torch.where( self.dof_pos[:, 0] > self._length_strait_softleg, (torch.atan(5 * torch.norm(self.dof_vel[:, 1:-1], dim=1)).view(-1, 1) / 1e4).squeeze(), 0)
        self.rew_buf[:] = - 100 * rew_vel_mio 
    
        rew_ground_2 = torch.where(((self.dof_pos[:,0] < self._length_strait_softleg) & (self.from_knee_to_foot < 0.015)) | 
                                   ((self.dof_pos[:,0] < self._length_strait_softleg) & (self.contact_force_foot[:,-1] > 0.) ), 0.1, 0)
        rev_pos_q1 = torch.where((self.dof_pos[:,1] < 0.0), 0.1, 0)
        rev_pos_q2 = torch.where((self.dof_pos[:,2] < 0.0), 0.1, 0)

        self.rew_buf[:] += 5 * rew_ground_2 + 10 * rev_pos_q1 + 10 * rev_pos_q2 - 100 * rew_action_rate

        self.dof_pos_save = torch.cat([self.dof_pos_save, self.dof_pos], dim=1) 
        self.dof_vel_save = torch.cat([self.dof_vel_save, self.dof_vel], dim=1) 
        self.actions_save = torch.cat([self.actions_save, self._torques], dim=1) 
        self.contact_force_foot_save = torch.cat([self.contact_force_foot_save, self.contact_force_foot], dim=1) 
        
        if (self.progress_buf * self._dt == self._max_episode_length_s).all():
            max_hight_per_softleg, _ = torch.max(self._high_softlegs, dim=1, keepdim=True)
            self.episode_sums["max_reached_height"] = max_hight_per_softleg
            err_reached = torch.norm(torch.stack([torch.tensor([self._height_des], device=self._device)] * self._num_envs, dim=0) - max_hight_per_softleg, dim=1)

            self.rew_buf[:] += 400 / (1 + err_reached ** 2) + 10 * rew_ground
            
            
            if self._cfg['test']:
                print('[INFO]: \t Max Height reached: \t{}'.format(max_hight_per_softleg.flatten()))
                # self.my_callback_testing('test_jump_10')
                self.my_callback_testing_each_env(self.filename_2save)
                self.my_plot_rn(self.dof_pos_save, self.dof_vel_save, self.actions_save, 'hip', self.epoch_num, self.filename_2save)
                self.my_plot_rn( self.dof_pos_save, self.dof_vel_save, self.actions_save, 'knee', self.epoch_num, self.filename_2save)
                
                self.epoch_num += 1
                if self._count % 10 == 0 and self.WANNA_INFO:
                    input('check prints ...')
        else:
            pass
        
    def is_done(self, toll=1e-5) -> None:
        check_time = self.progress_buf * self._dt >= self._max_episode_length_s
        self.reset_buf[:] = check_time
    
        
    def my_callback_testing(self, name_folder):
        '''
        This function has to be run during the test phase.
        This function has to be run at the end of the epoch.
        This function woks just for a fixed horizont of the epoch 
        '''
        
        assert isinstance(name_folder, str), "name_folder is not of type 'str'"
        
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
        else:
            pass
        
        epoch_folder = os.path.join(name_folder, str(self.epoch_num))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)            
        
        file_names = ["dof_pos.csv", "dof_vel.csv", "actions.csv", "ground.csv"]
        ## first n_env x 3 has to be deleted in all, init-thisagio
        # tensors = [self.dof_pos, self.dof_vel, self.actions, self.contact_force_foot]
        tensors = [self.dof_pos_save, self.dof_vel_save, self.actions_save, self.contact_force_foot_save]

        for tensor, file_name in zip(tensors, file_names):
            file_path = os.path.join(epoch_folder, file_name)
            with open( file_path, "w", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(tensor.to('cpu').detach().numpy())
                
    def my_callback_testing_each_env(self, name_folder):
        '''
        This function has to be run during the test phase.
        This function assumes that none of the epoch double the other in terms of time 
    
        '''
        
        assert isinstance(name_folder, str), "name_folder is not of type 'str'"
        
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
        else:
            pass
        
        epoch_folder = os.path.join(name_folder, str(self.epoch_num))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)   
            
        file_names = ["dof_pos.csv", "dof_vel.csv", "actions.csv", "ground.csv"]
        
        env_ids = self.reset_buf.nonzero(as_tuple=True)[0]
        for i in env_ids:
            env_folder = os.path.join(epoch_folder, 'env_' + str(i.item()))
            if not os.path.exists(env_folder):
                os.makedirs(env_folder)  
            ## first n_env x 3 has to be deleted in all, init-thisagio
            tensors = [self.dof_pos_save[i.item(),:], self.dof_vel_save[i.item(),:], \
                self.actions_save[i.item(),:], self.contact_force_foot_save[i.item(),:]]

            for tensor, file_name in zip(tensors, file_names):
                file_path = os.path.join(env_folder, file_name)
                with open( file_path, "w", newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(tensor.to('cpu').detach().numpy().ravel()) # 1D array
                    
    def my_plot_rn(self, pos_dofs, vel_dofs, actions, keys_data, epoch, path):
        from matplotlib import pyplot as plt
        from matplotlib import rc
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        plt.rcParams['text.usetex'] = True
        font_size = 24
        labelWidth = 3
        
        name_folder = os.path.join(path, 'plots')
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
        else:
            pass
        
        pos_dofs = pos_dofs.view(self._num_dof, self.progress_buf[0] + 1)
        vel_dofs = vel_dofs.view(self._num_dof, self.progress_buf[0] + 1)
        actions = actions.view(self._num_actions, self.progress_buf[0] + 1)
        
        n, N = pos_dofs.shape if pos_dofs.ndim == 2 else (pos_dofs.shape[0], 1)
        pos_dofs = pos_dofs.reshape((max(N, n), min(N, n)))
        
        n, N = vel_dofs.shape if pos_dofs.ndim == 2 else (vel_dofs.shape[0], 1)
        vel_dofs = vel_dofs.reshape((max(N, n), min(N, n)))
        
        n, N = actions.shape if pos_dofs.ndim == 2 else (actions.shape[0], 1)
        actions = actions.reshape((max(N, n), min(N, n)))

        labelsPlot = [r'Hip', r'Knee']
        if keys_data == 'hip':
            index = 0
        elif keys_data == 'knee':
            index = 1
        else:
            print('[INFO]:\t Select either <hip> or <knee>')
        
        timeTask = torch.linspace(0, self._max_episode_length_s, N).cpu()
        pos_dofs = pos_dofs.cpu()
        vel_dofs = vel_dofs.cpu()
        actions = actions.cpu()
        
        plt.figure(figsize=(18, 14)) 
        fig, axs = plt.subplots(1, 2, figsize=(16, 12))

        axs[0].plot(timeTask[0:(self.progress_buf[0] + 1) // 4], pos_dofs[0:(self.progress_buf[0] + 1) // 4, index + 1], linewidth=labelWidth, color='b', linestyle='solid', label=labelsPlot[index])
        axs[0].plot(timeTask[0:(self.progress_buf[0] + 1) // 4], actions[0:(self.progress_buf[0] + 1) // 4, index], linewidth=labelWidth, color='r', linestyle='solid', label='Policy')
        axs[0].set_xlabel(r'$\mathbf{Time\,\, [s]}$', fontsize=font_size)
        axs[0].set_ylabel(r'$\mathbf{Position\,\, [rad]}$', fontsize=font_size)
        axs[0].legend(fontsize=font_size)
        axs[0].grid()
        axs[0].tick_params(labelsize=font_size)
        
        axs[1].plot(timeTask[0:(self.progress_buf[0] + 1) // 4], \
            pos_dofs[0:(self.progress_buf[0] + 1) // 4, index + 1] - actions[0:(self.progress_buf[0] + 1) // 4, index], \
            linewidth=labelWidth, color='b', linestyle='solid', label=labelsPlot[index])
        axs[1].set_xlabel(r'$\mathbf{Time\,\, [s]}$', fontsize=font_size)
        axs[1].set_ylabel(r'$\mathbf{Err \,\, Position\,\, [rad]}$', fontsize=font_size)
        axs[1].legend(fontsize=font_size)
        axs[1].grid()
        axs[1].tick_params(labelsize=font_size)
        # axs[2].plot(timeTask, vel_dofs[:, index + 1], linestyle='solid', color='b', linewidth=labelWidth, label=labelsPlot[index])
        # axs[2].set_xlabel(r'$\mathbf{Time\,\, [s]}$', fontsize=font_size)
        # axs[2].set_ylabel(r'$\mathbf{Velocity\,\, [rad/s]}$', fontsize=font_size)
        # axs[2].legend(fontsize=font_size)
        # axs[2].grid()
        # axs[].tick_params(labelsize=font_size)
        plt.savefig(name_folder + '/{}_{}.svg'.format(epoch, labelsPlot[index]), format='svg')
        # plt.show(block=False)
        plt.tight_layout()
        # plt.pause(0.001) 
            
        
        
        
        
        
        

