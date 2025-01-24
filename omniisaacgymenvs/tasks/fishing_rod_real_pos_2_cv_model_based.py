from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.fishingrobot import FishingRod
from omniisaacgymenvs.robots.articulations.views.fishingrod_view import FishingRodView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.objects import DynamicSphere
from pxr import UsdPhysics
import torch
from .fishing_rod_lumped_param.try_planner_fishing_3_casadi import *
import numpy as np 

####################################################################################################################################
### To run this code
## decomment what it is needed in task_util.py, i.e.,
## from omniisaacgymenvs.tasks.fishing_rod_real_pos_2_cv_model_based import FishingRodTaskPosDueCVModelBased
## "FishingRodPosDueCVMB" : FishingRodTaskPosDueCVModelBased
####################################################################################################################################
## ISAAC_PYTHON -m venv tolopesca3 --system-site-package
## source tolopesca3/bin/activate
## source pyenvs.sh
## (tolopesca3) michele@michele-Dell-G16-7630:~/Documents/OmniIsaacGymEnvs/omniisaacgymenvs$ python scripts/rlgames_train.py task=FishingRodPosDueCVMB num_envs=2 experiment=FishingRodPos_X_009_pos_new_2_Kpiu_MB headless=True
####################################################################################################################################

class FishingRodTaskPosDueCVModelBased(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        n_joints=22,
        WANNA_INFO=False,
        tracking_Z=False, 
        WANNA_MASS_CHANGE=False
    ) -> None:
        
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self.WANNA_INFO = WANNA_INFO
        self.WANNA_MASS_CHANGE = WANNA_MASS_CHANGE
        self._task_cfg = sim_config.task_config
        self._n_actuators = self._task_cfg["env"]["numActions"]
        self._n_joints = n_joints
        self._n_state = 2 * self._n_joints
        self._device = 'cuda:0'
        self._count = 0
        self.epoch_num = 0
        self.epoch_num_save = 0
        self.epoch_num_save_train = 0
        self.PRINT_INT = self._task_cfg["env"]["printInt"]
        self._num_observations = self._task_cfg["env"]["numObservations"]
        self.tracking_Z_bool = tracking_Z
        self._num_actions = self._n_actuators
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"] 
        self.gravity = torch.tensor(self._task_cfg["sim"]["gravity"][2], device=self._device)
        self.WANNA_MODEL_BASED_HELP_DUMMY = False
        amp = self._task_cfg["env"]["initState"]["ampK"] 
        k_ii = 1.5 * amp * torch.tensor([0.0, 34.61, 30.61, 26.84, 17.203, 11.9, 10.99,
                        12.61, 8.88, 4.04, 3.65, 3.05, 5.4, 3.67, 2.9, 3.02, 2.13, 1.6, 1.37, 1.01, 0.81, 0.6])
        d_ii = 2.0 * amp * torch.tensor([0.0, 0.191, 0.164, 0.127, 0.082, 0.056, 0.043, 0.060, 0.042, 0.019,
                        0.017, 0.015, 0.020, 0.017, 0.015, 0.014, 0.011, 0.009, 0.007, 0.003, 0.003, 0.003])
        self.d_ii_vect = d_ii.to(self._device)
        self.k_ii_vect = k_ii.to(self._device)
        self.D_matrix = torch.diag(d_ii) + torch.diag(d_ii[:-1] / 2e1, diagonal=-1) + torch.diag(d_ii[:-1] / 2e1, diagonal=1) + torch.diag(d_ii[:-2] / 2e1, diagonal=-2) + torch.diag(d_ii[:-2] / 2e1, diagonal=2)
        # self.D_matrix = torch.diag(d_ii) + torch.diag(d_ii[:-1] / 2e1, diagonal=-1) + torch.diag(d_ii[:-1] / 2e1, diagonal=1) + torch.diag(d_ii[:-2] / 5e1, diagonal=-2) + torch.diag(d_ii[:-2] / 5e1, diagonal=2)
        self.K_matrix = torch.diag(k_ii) + torch.diag(k_ii[:-1] / 2e1, diagonal=-1) + torch.diag(k_ii[:-1] / 2e1, diagonal=1) + torch.diag(k_ii[:-2] / 5e1, diagonal=-2) + torch.diag(k_ii[:-2] / 5e1, diagonal=2)
        self.D_matrix = self.D_matrix.clone().detach().to(dtype=torch.float, device=self._device).requires_grad_(False)
        self.K_matrix = self.K_matrix.clone().detach().to(dtype=torch.float, device=self._device).requires_grad_(False)
       
        self.torques_to_print = torch.zeros(self._num_envs, self._n_actuators, dtype=torch.float, device=self._device, requires_grad=False)
        self.torch_deg_sat = torch.deg2rad(torch.tensor(-self._task_cfg["env"]["initState"]["saveAngle"])).to(self._device) #  self._task_cfg["env"]["learn"]["qScale"]
        ## target ball features 
        self._ball_radius = 0.05
        self._ball_position = torch.zeros(self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._when_to_switch = self._task_cfg["env"]["switchCV"] # number of epoch afert which I try to set both the position and the velocity
        if self._cfg["test"]:
            import os 
            self.filename_2save = os.path.splitext(os.path.basename(self._cfg['checkpoint']))[0]
            
        self._q_scale = self._task_cfg["env"]["learn"]["qScale"]
        self._q_dot_scale = self._task_cfg["env"]["learn"]["qDotScale"]
        self._action_scale = self._task_cfg["env"]["control"]["actionScale"]
        self._when_to_print = self._task_cfg["env"]["whenToPrint"] 
        self._length_fishing_rod = self._task_cfg["env"]["lengthFishingRod"]
        self._length_fishing_rod_y = self._task_cfg["env"]["lengthFishingY"]
        self.control_frequency_inv_real = self._task_cfg["env"]["controlFrequencyInvReal"]
        
        self._length_fishing_rod = torch.tensor([self._length_fishing_rod] * self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)
        self._length_fishing_rod_y = torch.tensor([self._length_fishing_rod_y] * self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)
        
        self._vel_lin_scale = self._task_cfg["env"]["learn"]["velLinTipScale"]
        self._vel_or_scale = self._task_cfg["env"]["learn"]["velOriTipScale"]
        self._pos_scale = self._task_cfg["env"]["learn"]["positionTipScale"]
        self._or_scale = self._task_cfg["env"]["learn"]["orientationTipScale"]
        self._task_cfg["sim"]["add_ground_plane"] = True
    
        ## initial config 
        self.q_init = self._task_cfg["env"]["initState"]["q"]
        self.q_dot_init = self._task_cfg["env"]["initState"]["qDot"]
        self._reset_dist = self._task_cfg["env"]["resetDist"] 
        self._max_effort = self._task_cfg["env"]["maxEffort"]
        self._max_obs = self._task_cfg["env"]["clipObservations"]
        self._decimation = self._task_cfg["env"]["control"]["decimation"]
        self._sim_dt = self._task_cfg["sim"]["dt"]
        self._dt = self._decimation * self._sim_dt
        self._max_episode_length_s = self._task_cfg["env"]["episodeLength"] 
        self._init_state = self._task_cfg["env"]["initState"]["q"]
        self.noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        self._acc_scale = self._task_cfg["env"]["learn"]["accLinTipScale"]
        self._acc_noise = self._task_cfg["env"]["learn"]["accLinTipNoise"]
                
        self.rew_scales = {}        
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self._dt
            
        ## low level control gains 
        self._Kp = torch.tensor(self._task_cfg["env"]["control"]["stiffness"], dtype=torch.float, device=self._device, requires_grad=False)
        self._Kd = torch.tensor(self._task_cfg["env"]["control"]["damping"], dtype=torch.float, device=self._device, requires_grad=False)
        
        self.extras = {}
        self.print_one = True
        self._last_config = torch.zeros(self._num_envs, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        
        RLTask.__init__(self, name, env)  

        self._noise_scale_vec = self.get_noise_scale_vec(self._task_cfg)
        
        self.torques = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self.actions = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._last_actions = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._last_joint_vel = torch.zeros(self._num_envs, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        torch_zeros = lambda : torch.zeros(self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)
        self.torques_all = torch.zeros(self._num_envs, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        
        self.tip_acc_lin = torch.zeros(self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)
        self.tip_acc_lin_old = torch.zeros_like(self.tip_acc_lin)
        self.tip_vel_lin_old = torch.zeros_like(self.tip_acc_lin)
        self.tip_pos_old = torch.zeros_like(self.tip_acc_lin)
        
        self.episode_sums = {"err_pos": torch_zeros(), 
                             "torque": torch_zeros(), 
                             "err_vel": torch_zeros(),
                             "action_rate": torch_zeros(), 
                             "velocity_final": torch_zeros(), 
                             "joint_acc": torch_zeros(), 
                             "joint_vel": torch_zeros(), 
                             "joint_pos": torch_zeros()}
        
        ## desired task 
        self.min_vel_lin_des, self.max_vel_lin_des = self._task_cfg["env"]["task"]["minVelDes"], self._task_cfg["env"]["task"]["maxVelDes"] # [m/s] only one component
        self._min_mass_tip, self._max_mass_tip = self._task_cfg["env"]["task"]["minMassTip"], self._task_cfg["env"]["task"]["maxMassTip"] # [Kg]
        self.new_mass = self._min_mass_tip + (self._max_mass_tip - self._min_mass_tip) * torch.rand(self._num_envs, device=self._device)
        if self.tracking_Z_bool:
            self.min_pos_des, self.max_pos_des = self._task_cfg["env"]["task"]["minPosDesZ"], self._task_cfg["env"]["task"]["maxPosDesZ"]   # Z-component 
            self._pos_des = self._length_fishing_rod - torch.clamp((self.max_pos_des - self.min_pos_des) * torch.rand((self._num_envs,), dtype=torch.float, device=self._device) + self.min_pos_des, self.min_pos_des, self.max_pos_des)
        else:
        ## tracking X
            self.min_pos_des, self.max_pos_des = self._task_cfg["env"]["task"]["minPosDesX"], self._task_cfg["env"]["task"]["maxPosDesX"]
            self._pos_des = torch.clamp((self.max_pos_des - self.min_pos_des) * torch.rand((self._num_envs,), dtype=torch.float, device=self._device) + self.min_pos_des, self.min_pos_des, self.max_pos_des)
        
        self._vel_lin_des = (self.max_vel_lin_des - self.min_vel_lin_des) * torch.rand((self._num_envs,), dtype=torch.float, device=self._device) + self.min_vel_lin_des
        self._vel_lin_des = -self._vel_lin_des
        self.des_y_coordinate = torch.sqrt(self._length_fishing_rod**2 - self._pos_des**2)
        
        pos_des_np_mean = torch.mean(self._pos_des).cpu().numpy() 
        vel_lin_des_np_mean = torch.mean(self._vel_lin_des).cpu().numpy() 
        des_y_coordinate_np_mean = torch.mean(self.des_y_coordinate).cpu().numpy()
        
        if self.tracking_Z_bool:
            ## pos_d = [Z, X]
            pos_d = np.array([des_y_coordinate_np_mean, pos_des_np_mean])
        else:
            pos_d = np.array([pos_des_np_mean, des_y_coordinate_np_mean])
        # vel_d = -vel_lin_des_np_mean
        vel_d = vel_lin_des_np_mean
        
        print(f'[INFO] Desired Position -> Opt : {pos_d}')
        print(f'[INFO] Desired Velocity -> Opt : {np.round(vel_d, 2)}')
        
        self.u_model_based = main_fun_optmial_casadi(tracking_Z_bool=self.tracking_Z_bool, 
                                                    pos_d=pos_d, 
                                                    _max_episode_length_s=self._max_episode_length_s,
                                                    vel_des=vel_d)
        # self.u_model_based = main_fun_optmial_casadi()
        
        self.u_model_based = -np.array(self.u_model_based) # (_n_envs, time)
        # self.u_model_based = np.array(self.u_model_based) # (_n_envs, time)
        self.u_model_based_torch = torch.tensor(self.u_model_based, dtype=torch.float, device=self._device).view(1, -1).repeat(self._num_envs, 1)
                      
        print('\n')
        print('=' * self.PRINT_INT)
        if self.min_pos_des != self.max_pos_des:
            print('[INFO]: Desired Positions is changing any epoch')
            if ~self.tracking_Z_bool:
                print('[INFO]: The point is simmetric w.r.t. the fishing rod')
        if self.min_vel_lin_des != self.max_vel_lin_des:
            print(f'[INFO]: Desired Velocity is changing any epoch after {self._when_to_switch} epochs')
        print('=' * self.PRINT_INT)
        print('\n\n')
        return
            
    def generate_trajectory(self, time, half_period=0.5, scale=0.1):
        '''Generate a sinusoidal trajectory with a given half period and scale.
        Suppose to help the first joint within the motion.'''
        trajectory = scale * (1 - torch.cos((2 * torch.pi / half_period) * time))        
        return trajectory
    
    def get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self._add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        # noise_vec[0] = 0 ## no noise action
        noise_vec[0] = self._task_cfg["env"]["learn"]["qNoise"] * self.noise_level * self._action_scale 
        noise_vec[1] = self._task_cfg["env"]["learn"]["qDotNoise"] * self.noise_level * self._q_dot_scale 
        noise_vec[2] = self._task_cfg["env"]["learn"]["velLinTipNoise"] * self.noise_level * self._vel_lin_scale
        noise_vec[3] = self._task_cfg["env"]["learn"]["posTipNoise"] * self.noise_level * self._pos_scale
        noise_vec[4] = self._task_cfg["env"]["learn"]["posTipNoise"] * self.noise_level * self._pos_scale
        noise_vec[5] = self._task_cfg["env"]["learn"]["velLinTipNoise"] * self.noise_level * self._vel_lin_scale
        noise_vec[6] = 0
        noise_vec[7] = 0  
        noise_vec[8] = self._task_cfg["env"]["learn"]["accLinTipNoise"] * self.noise_level * self._acc_scale
        noise_vec[9] = self._task_cfg["env"]["learn"]["accLinTipNoise"] * self.noise_level * self._acc_scale
        noise_vec[-1] = 0 # time
    
        return noise_vec
    
    def set_up_scene(self, scene) -> None:
        
        self._stage = get_current_stage() 
        self.get_fishingrod()
        super().set_up_scene(scene)
        self._fishingrods = FishingRodView(prim_paths_expr="/World/envs/.*/fishingrod", 
                                        name="FishingRodView", 
                                        default_dof_pos = self.q_init,
                                        track_contact_forces=False) 
        scene.add(self._fishingrods)
        scene.add(self._fishingrods._tip)
        scene.add(self._fishingrods._base)
        if self.WANNA_MASS_CHANGE:
            self.update_tip_mass_all()
    
    def update_tip_mass_all(self):
        for i in range(self._num_envs):
            # for env_id in env_ids:
            tip_path = f'/World/envs/env_{i}/fishingrod/Tip'
            tip_prim = self._stage.GetPrimAtPath(tip_path)
            if not tip_prim:
                print(f"Tip link not found in environment {i}")
                continue
            mass_api = UsdPhysics.MassAPI(tip_prim)
            if mass_api:
                mass_attr = mass_api.GetMassAttr()
                if not mass_attr:
                    mass_attr = mass_api.CreateMassAttr()
                mass_attr.Set(self.new_mass[i].item())
            else:
                pass

    def get_fishingrod(self):
        ## The fishing rod appears to be in [0.0000, 0.0700, 2.9417]
        self._init_state = torch.tensor(self._init_state, dtype=torch.float, device=self._device, requires_grad=False)
        fishingrod = FishingRod(prim_path=self.default_zero_env_path + "/fishingrod", name="FishingRod")
        self._sim_config.apply_articulation_settings("FishingRod",
                                                    get_prim_at_path(fishingrod.prim_path),
                                                    self._sim_config.parse_actor_config("FishingRod"))
        if self.WANNA_INFO and self._cfg["test"] and self._cfg["livestream"]:
            print('\n')
            print('=' * self.PRINT_INT)
            print(dir(fishingrod))
            input('HHHHHHalmaaaaaa')
            
        self.dof_names = fishingrod.dof_names    
        self._name_joint = ['fishing_actuator_joint'] + [f'Joint_{i}' for i in range(1, self._n_joints)]
       
    def refresh_dof_state_tensors(self):
        
        self.dof_pos = self._fishingrods.get_joint_positions(clone=False)
        self.dof_vel = self._fishingrods.get_joint_velocities(clone=False)
        # self.tip_pos, self.tip_or = self._fishingrods._tip.get_local_poses() ## very slow :( 
        self.tip_pos, self.tip_or = self._fishingrods._tip.get_world_poses()
        self.base_pos, self.base_or = self._fishingrods._base.get_world_poses()
        self.base_vel_init = self._fishingrods._base.get_linear_velocities(clone=False)

        if self.tracking_Z_bool:
            pass 
        else:
            self.tip_pos[:, 0] = self.tip_pos[:, 0] - self._env_pos[:, 0]
            # self.tip_pos[:, 0] = self.tip_pos[:, 0] - self.base_pos[:, 0]
            self.tip_vel_lin[:, 0] = self.tip_vel_lin[:, 0] - self.base_vel_init[:, 0] 
    
        self.tip_vel_or = self._fishingrods._tip.get_angular_velocities(clone=False)
        
        # self.tip_vel_lin = self._fishingrods._tip.get_linear_velocities(clone=False)  
        ## this is more realistic but it does not work properly in the learning 
        self.tip_vel_lin = self.tip_pos - self.tip_pos_old / self._dt
        
        if self._cfg["test"] or self._cfg["livestream"]:
            for i in range(0, self._num_envs):
                self._ball_position[i,:] = self.tip_pos[i,:] 
                if self.tracking_Z_bool:
                    self._ball_position[i, -1] = self._pos_des[i]
                else:
                    self._ball_position[i, 0] = self._pos_des[i]
            self.get_target() 
        
        if self.WANNA_INFO and self._count%self._when_to_print == 0:
            if (self.progress_buf * self._dt == self._max_episode_length_s).all() and self.WANNA_INFO and ~self._cfg["test"]:
                print('[INFO] Stop episode ...')
                input('=' * self.PRINT_INT)
            if self.tracking_Z_bool:
                print('[INFO] Pos Z   : ', self.tip_pos[:,-1].flatten())
            else:
                print('[INFO] Pos X   : ', self.tip_pos[:,0].flatten())
            print('[INFO] Pos Des : ', self._pos_des.flatten())
            print('[INFO] Vel L X : ', self.tip_vel_lin[:, 0].flatten())
            print('[INFO] Vel Des : ', self._vel_lin_des.flatten())
            print('[INFO] Action  : ', self.torques_to_print.flatten())
            print('[INFO] Torque  : ', (self._action_scale * self.actions[:, 0]).flatten())
            print('[INFO] Time    : ', self._count * self._dt)
            print('=' * self.PRINT_INT)
    
    def get_observations(self) -> dict: 
                
        # self.refresh_dof_state_tensors()
        if self.WANNA_INFO:
            print(f"[INFO] tip pos     : {self.tip_pos[:3, 0]}")
            print(f"[INFO] tip pos old : {self.tip_pos_old[:3, 0]}")
            print('\n')
            print(f'[INFO] vel tip     : {(self.tip_pos[:3, 0] - self.tip_pos_old[:3, 0] / self._dt)}')
            print(f'[INFO] vel isaac   : {(self.tip_vel_lin[:3, 0])}')
            print('\n')
            print(f'[INFO] acc tip     : {self.tip_acc_lin[:3]}')
            print(f'[INFO] acc tip old : {self.tip_acc_lin_old[:3]}')
            print('-' * 50)
        self.tip_acc_lin = (self.tip_vel_lin[:, 0] - self.tip_vel_lin_old[:, 0]) / self._dt
        
        self.obs_buf[:, 0] = self.actions[:, 0] 
        self.obs_buf[:, 1] = self.dof_vel[:, 0] / self._q_dot_scale
        
        if self.tracking_Z_bool:
            self.obs_buf[:, 2] = self.tip_vel_lin[:, -1] / self._vel_lin_scale
            self.obs_buf[:, 3] = self.tip_pos[:, -1] / self._pos_scale
            self.obs_buf[:, 4] = self.tip_pos_old[:, -1] / self._pos_scale
            self.obs_buf[:, 5] = self.tip_vel_lin_old[:, -1] / self._vel_lin_scale
        else:
            self.obs_buf[:, 2] = self.tip_vel_lin[:, 0] / self._vel_lin_scale
            self.obs_buf[:, 3] = self.tip_pos[:, 0] / self._pos_scale
            self.obs_buf[:, 4] = self.tip_pos_old[:, 0] / self._pos_scale
            self.obs_buf[:, 5] = self.tip_vel_lin_old[:, 0] / self._vel_lin_scale
                        
        self.obs_buf[:, 6] = self._err_vel_all[:, self.progress_buf[0] - 1]
        self.obs_buf[:, 7] = self._err_pos_all[:, self.progress_buf[0] - 1] 
        self.obs_buf[:, 8] = self.tip_acc_lin[:] / self._acc_scale
        self.obs_buf[:, 9] = self.tip_acc_lin_old[:] / self._acc_scale
        
        self.obs_buf[:, 10] = self.u_model_based_torch[:, self.progress_buf[0] - 1] / self._action_scale
        self.obs_buf[:, -1] = (self._max_episode_length_s - self._count * self._dt) / self._max_episode_length_s
        
        observations = {self._fishingrods.name: {"obs_buf": self.obs_buf}}
        
        return observations 
    
    def pre_physics_step(self, actions) -> None:
     
        if not self._env._world.is_playing():
            return
        
        self.actions[:,:] = actions.clone().to(self._device) 
        help_term = torch.zeros(self._num_envs, dtype=torch.float, device=self._device)
        for _ in range(self._decimation):  
            if self._env._world.is_playing():   
                
                if self.WANNA_MODEL_BASED_HELP_DUMMY:
                    help_term = self.generate_trajectory(self.progress_buf[0] * self._dt) * torch.ones(self._num_envs, dtype=torch.float, device=self._device)
                else:
                    pass
                
                torques = torch.clip( self._Kp * ( self._action_scale * self.actions[:, 0] - self.dof_pos[:, 0] + help_term) \
                    + self.u_model_based_torch[:, self.progress_buf[0] - 1] - self._Kd * self.dof_vel[:, 0], -self._max_effort, self._max_effort)
                
                self.torques_to_print = torques
                self.torques_all[:, 0] = torques
                torques = self.torques_all
                stiffness_torque = torch.matmul(self.dof_pos, self.K_matrix.t()).detach().requires_grad_(True).to(dtype=torch.float, device=self._device)
                damping_torque = torch.matmul(self.dof_vel, self.D_matrix.t()).detach().requires_grad_(True).to(dtype=torch.float, device=self._device)
                friction_air_torque = ( self.noise_level / 1e1 ) * ( 2 * torch.rand(self._num_envs, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False) - 1) \
                    * self.d_ii_vect.unsqueeze(0).expand(self._num_envs, -1) # eventually put directly stiffness_torque or damping torque
                    
                self.torques = torques - stiffness_torque - damping_torque + friction_air_torque                
                self._fishingrods.set_joint_efforts(self.torques)
                
                if self._count%self._when_to_print and self.WANNA_INFO and self._cfg["test"] == 0:
                    print('[INFO] friction_air_torque : ', friction_air_torque.flatten())
                    print('[INFO] stiffness_torque    : ', stiffness_torque.flatten())
                    print('[INFO] damping_torque      : ', damping_torque.flatten())
                    print('[INFO] torque              : ', self.torques.flatten())
                    input('=' * self.PRINT_INT)
                
                for _ in range(self.control_frequency_inv_real):
                    self._env._world.step(render=self._env._render)
                    self._env.sim_frame_count += 1
                    # SimulationContext.step(self._env._world, render=False)
                    
                self.tip_vel_lin_old = self.tip_vel_lin.clone()
                self.tip_pos_old = self.tip_pos.clone()
                self.tip_acc_lin_old = self.tip_acc_lin.clone()
                self.action_old = self.actions[:]
                self.refresh_dof_state_tensors()
        
    def get_target(self):
        
        for i in range(0, self._num_envs):
            ball = DynamicSphere(
                prim_path=self.default_zero_env_path + "/ball",
                translation=self._ball_position[i, :],
                name="target_" + str(i),
                radius=self._ball_radius,
                color=torch.tensor([1, 0, 0]))
            self._sim_config.apply_articulation_settings("ball", 
                                                        get_prim_at_path(ball.prim_path),
                                                        self._sim_config.parse_actor_config("ball"))
            ball.set_collision_enabled(False)
        
    def reset_idx(self, env_ids):
     
        num_resets = len(env_ids)
        self._count = 0

        self.dof_pos[env_ids, :] = torch.zeros(num_resets, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        self.dof_vel[env_ids, :] = torch.zeros(num_resets, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        self.torques_all[env_ids, :] = torch.zeros(num_resets, self._n_joints, dtype=torch.float, device=self._device, requires_grad=False)
        indices = env_ids.to(dtype=torch.int32)
        self.des_y_coordinate[env_ids] = torch.sqrt(self._length_fishing_rod**2 - self._pos_des[env_ids]**2)

        self._err_vel_all = torch.zeros(self._num_envs, int(self._max_episode_length_s/self._dt), dtype=torch.float, device=self._device, requires_grad=False)
        self._err_pos_all = torch.zeros(self._num_envs, int(self._max_episode_length_s/self._dt), dtype=torch.float, device=self._device, requires_grad=False)
        
        self._fishingrods.set_joint_positions(self.dof_pos, indices=indices)
        self._fishingrods.set_joint_velocities(self.dof_vel, indices=indices)

        if self.tracking_Z_bool:
            self._pos_des[env_ids] = self._length_fishing_rod - torch.clamp((self.max_pos_des - self.min_pos_des) * torch.rand((num_resets,), dtype=torch.float, device=self._device) + self.min_pos_des, self.min_pos_des, self.max_pos_des)
        else:
            self._pos_des[env_ids] = (self.max_pos_des - self.min_pos_des) * torch.rand((num_resets,), dtype=torch.float, device=self._device) + self.min_pos_des
        
        if self.epoch_num > self._when_to_switch or self._cfg["test"]:
            ## curriculum learning
            self._vel_lin_des[env_ids] = (self.max_vel_lin_des - self.min_vel_lin_des) * torch.rand((num_resets,), dtype=torch.float, device=self._device) + self.min_vel_lin_des
        else:
            self._vel_lin_des[env_ids] =  (( self.max_vel_lin_des + self.min_vel_lin_des) / 2) * torch.ones((num_resets,), dtype=torch.float, device=self._device) 
        
        self._vel_lin_des[env_ids] = -self._vel_lin_des[env_ids] 

        self.dof_pos_save = self.dof_pos
        self.dof_vel_save = self.dof_vel
        self.actions_save = self.actions
        self.tip_pos_save = torch.zeros(num_resets, 3, dtype=torch.float, device=self._device)
        self.tip_vel_save = torch.zeros(num_resets, 3, dtype=torch.float, device=self._device)
        self._pos_des_save_3D = torch.zeros(num_resets, 3, dtype=torch.float, device=self._device)
        self._vel_lin_des_save = torch.zeros((num_resets, 0), dtype=torch.float, device=self._device)
        self.torque_save = torch.zeros(num_resets, self._n_actuators, dtype=torch.float, device=self._device)
        
        if (self.epoch_num_save_train > 2) and (self.progress_buf[:]).all() > 1:    ## never doing this 
        
            if self.tracking_Z_bool:
                ## pos_d = [Z, X]
                pos_d = np.array([torch.mean(self.des_y_coordinate[env_ids]).cpu().numpy(), torch.mean(self._pos_des[env_ids]).cpu().numpy()])
            else:
                pos_d = np.array([torch.mean(self._pos_des[env_ids]).cpu().numpy(), torch.mean(self.des_y_coordinate[env_ids]).cpu().numpy()])
            vel_d = -torch.mean(self._vel_lin_des[env_ids]).cpu().numpy()
            
            self.u_model_based = main_fun_optmial_casadi(tracking_Z_bool=self.tracking_Z_bool, 
                                                        pos_d=pos_d, 
                                                        _max_episode_length_s=self._max_episode_length_s,
                                                        vel_des=vel_d)
            
            self.u_model_based = -np.array(self.u_model_based) # (_n_envs, time)
            self.u_model_based_torch[env_ids] = torch.tensor(self.u_model_based, dtype=torch.float, device=self._device).view(1, -1).repeat(env_ids.size(0), 1)
            
        indices = env_ids.to(dtype=torch.int32)
         
        self._fishingrods.set_joint_positions(self.dof_pos, indices=indices)
        self._fishingrods.set_joint_velocities(self.dof_vel, indices=indices)
        
        self._env._world.step(render=self._env._render)
        self.refresh_dof_state_tensors() # check this one
        if self.WANNA_MASS_CHANGE:
            self.update_tip_mass_all()
        
        # bookkeeping
        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0
        self._last_actions[env_ids] = 0.     
        self._last_config[env_ids] = 0.
        self._last_joint_vel[env_ids] = 0.
        self.epoch_num_save_train += 1
                
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0  
            
    def post_reset(self):
        self._num_dof = self._n_joints
        self.dof_pos = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self.dof_vel = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self.actions = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self.tip_pos = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.tip_or = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device, requires_grad=False)
        self.tip_vel_lin = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.tip_vel_or = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device, requires_grad=False)
        self.tip_pos_old = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.tip_vel_lin_old = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.tip_acc_lin_old = torch.zeros(self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)
        self.action_old = torch.zeros((self._num_envs, self._num_actions), dtype=torch.float, device=self._device, requires_grad=False)
        # self._pos_des = torch.zeros(self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)
        # self._vel_lin_des = torch.zeros(self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)

        indices = torch.arange(self._fishingrods.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def post_physics_step(self):
        self.progress_buf[:] += 1           
            
        if self._env._world.is_playing():
            self.refresh_dof_state_tensors()
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
        
        self._count += 1
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def calculate_metrics(self) -> None:
        
        rew_action_rate = torch.sum(torch.square(self.actions), dim=-1) 
        rew_velocity = torch.sum(torch.square(self.dof_vel), dim=-1) 
        self.episode_sums["joint_vel"] += rew_velocity
        self.episode_sums["action_rate"] += rew_action_rate
                
        if self.tracking_Z_bool:
            err_reached_pos = self._pos_des - self.tip_pos[:, -1]
        else:
            err_reached_pos = self._pos_des - self.tip_pos[:, 0] 
        
        module_vel = self.tip_vel_lin[:, 0] ## only on the X-component for the velocity             
        err_reached_vel = self._vel_lin_des - self.tip_vel_lin[:, 0]

        self._err_vel_all[:, self.progress_buf[0] - 1] = err_reached_vel
        self._err_pos_all[:, self.progress_buf[0] - 1] = err_reached_pos
        
        if self._cfg["test"]:
            self.dof_pos_save = torch.cat([self.dof_pos_save, self.dof_pos], dim=1) 
            self.dof_vel_save = torch.cat([self.dof_vel_save, self.dof_vel], dim=1) 
            self.actions_save = torch.cat([self.actions_save, self.actions], dim=1) 
            self.tip_pos_save = torch.cat([self.tip_pos_save, self.tip_pos], dim=1) 
            self.tip_vel_save = torch.cat([self.tip_vel_save, self.tip_vel_lin], dim=1) 
            self._pos_des_save_3D = torch.cat([self._pos_des_save_3D, self._ball_position], dim=1) 
            self._vel_lin_des_save = torch.cat([self._vel_lin_des_save, self._vel_lin_des.unsqueeze(1)], dim=1)  
            self.torque_save = torch.cat([self.torque_save, self.torques_to_print.unsqueeze(-1)], dim=1)            
                  
        if (self.progress_buf * self._dt >= self._max_episode_length_s).all():
            self.epoch_num += 1
            self.episode_sums["err_pos"] = err_reached_pos[:]
            self.episode_sums["err_vel"] = err_reached_vel[:]
            self.episode_sums["velocity_final"] = self.tip_vel_lin[:]
            self.episode_sums["joint_pos"] = self.dof_pos[:] 
            
            self.rew_buf[:] = ( 5 / (1 + err_reached_pos ** 2) + 2 / (1 + err_reached_vel ** 2) ) * self._max_episode_length_s \
                                + torch.where(module_vel < 0, 5 / (1 + err_reached_vel ** 2), -5 / (1 + err_reached_vel ** 2)) * self._max_episode_length_s
                                            
            if self._cfg['test']:
                print('\n')
                print('=' * self.PRINT_INT)
                if self.print_one:
                    print('[INFO] Checking, at the last time instant, the condition of the tip')
                    print('=' * self.PRINT_INT)
                    self.print_one = False
                if self.tracking_Z_bool:
                    print('[INFO] Pos Tip :  ', ((self.tip_pos[:,-1]).flatten()))
                else:
                    print('[INFO] Pos Tip :  ', torch.abs((self.tip_pos[:,0]).flatten()))
                
                print('[INFO] Pos Des :  ', ((self._pos_des).flatten()))
                print('=' * self.PRINT_INT)
                print('[INFO] Vel Tip :  ', (module_vel.flatten()))
                print('[INFO] Vel Des :  ', (self._vel_lin_des.flatten()))
                print('=' * self.PRINT_INT)
                print('[INFO] Err Pos :  ', (err_reached_pos.flatten()))
                print('[INFO] Err Vel :  ', (err_reached_vel.flatten()))
                print('=' * self.PRINT_INT)
                input('[INFO] check ...')
                print('=' * self.PRINT_INT)
                self.my_callback_testing_each_env(self.filename_2save)
                self.epoch_num_save += 1
                if self._count % 10 == 0 and self.WANNA_INFO and self._cfg["test"]:
                    input('Check prints ...')   
        else:
            pass
        
    def is_done(self, toll=1e-5) -> None:
        check_time = self.progress_buf * self._dt >= self._max_episode_length_s
        self.reset_buf[:] = check_time
        
    def my_callback_testing_each_env(self, name_folder):
        '''
        This function has to be run during the test phase.
        This function assumes that none of the epoch double the other in terms of time 
        '''
        import os 
        import csv 
        assert isinstance(name_folder, str), "name_folder is not of type 'str'"
        
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
        # else:
        #     # name_folder = name_folder + '_' + str( str(int(torch.floor(torch.rand(1) * 100))))
        #     print('The folder "{}" already exists!'.format(name_folder))
        #     input('[INFO] Press <ENTER> to continue ...')

        D_matrix_cpu, K_matrix_cpu = self.D_matrix.cpu().numpy(), self.K_matrix.cpu().numpy()
        D_matrix_file, K_matrix_file = os.path.join(name_folder, 'D_matrix.txt'), os.path.join(name_folder, 'K_matrix.txt')

        with open(D_matrix_file, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in D_matrix_cpu:
                csvwriter.writerow(row.tolist())
        with open(K_matrix_file, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in K_matrix_cpu:
                csvwriter.writerow(row.tolist())
                
        if self.WANNA_MASS_CHANGE:        
            if self.new_mass is not None:
                new_mass_cpu = self.new_mass.cpu().numpy()
                new_mass_file = os.path.join(name_folder, 'new_mass.txt')
                with open(new_mass_file, "w", newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(new_mass_cpu.tolist())
        
        epoch_folder = os.path.join(name_folder, str(self.epoch_num_save))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)        
        
        file_names = ["dof_pos.csv", "dof_vel.csv", "actions.csv", "tip_pos.csv", 
                    "tip_vel_save.csv", "pos_des_save_3D.csv", "vel_lin_des.csv", 'torque.csv']
        
        env_ids = self.reset_buf.nonzero(as_tuple=True)[0]
        for i in env_ids:
            env_folder = os.path.join(epoch_folder, 'env_' + str(i.item()))
            if not os.path.exists(env_folder):
                os.makedirs(env_folder)  
            
            ## first n_env x 3 has to be deleted in all, init-thisagio
            tensors = [self.dof_pos_save[i.item(),:], self.dof_vel_save[i.item(),:], \
                    self.actions_save[i.item(),:], self.tip_pos_save[i.item(),:], \
                    self.tip_vel_save[i.item(),:], self._pos_des_save_3D[i.item(),:], \
                    self._vel_lin_des_save[i.item(),:], self.torque_save[i.item(),:]]
            
            self.my_plot_rn(self.tip_pos_save[i.item(),:], self.tip_vel_save[i.item(),:], 
                            self.actions_save[i.item(),:], self.torque_save[i.item(),:], 
                            self.epoch_num_save, env_folder)

            for tensor, file_name in zip(tensors, file_names):
                file_path = os.path.join(env_folder, file_name)
                with open( file_path, "w", newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(tensor.to('cpu').detach().numpy().ravel()) # 1D array
                    
        print('[INFO] Plotting and Saving ...')
        print('=' * self.PRINT_INT)
                    
    def my_plot_rn(self, pos_tip, vel_tip, actions, torque, epoch, path):
        from matplotlib import pyplot as plt
        from matplotlib import rc
        import os 

        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        plt.rcParams['text.usetex'] = True

        font_size, labelWidth = 24, 5
        size_1, size_2 = 9, 7

        name_folder = os.path.join(path, 'plots')
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

        device = pos_tip.device
        pos_tip = pos_tip.view(3, self.progress_buf[0] + 1).to(device)
        vel_tip = vel_tip.view(3, self.progress_buf[0] + 1).to(device)
        actions = actions.view(1, self.progress_buf[0] + 1).to(device)
        torque = torque.view(1, self.progress_buf[0] + 1).to(device)
        
        n, N = pos_tip.shape if pos_tip.ndim == 2 else (pos_tip.shape[0], 1)
        pos_tip = pos_tip.reshape((max(N, n), min(N, n)))

        n, N = vel_tip.shape if vel_tip.ndim == 2 else (vel_tip.shape[0], 1)
        vel_tip = vel_tip.reshape((max(N, n), min(N, n)))

        timeTask = torch.linspace(0, self._max_episode_length_s, N).to(device)
        
        if self._pos_des.ndim == 1:
            pos_des = self._pos_des.unsqueeze(1).to(device) * torch.ones_like(timeTask)
        elif self._pos_des.ndim == 2:
            pos_des = self._pos_des.to(device) * torch.ones_like(timeTask)
        else:
            raise ValueError("Unexpected _pos_des dimensions")

        if self._vel_lin_des.ndim == 1:
            vel_des = self._vel_lin_des.unsqueeze(1).to(device) * torch.ones_like(timeTask)
        elif self._vel_lin_des.ndim == 2:
            vel_des = self._vel_lin_des.to(device) * torch.ones_like(timeTask)
        else:
            raise ValueError("Unexpected _vel_lin_des dimensions")

        pos_tip, vel_tip = pos_tip.cpu(), vel_tip.cpu()
        pos_des, vel_des = pos_des.cpu(), vel_des.cpu()
        timeTask, actions = timeTask.cpu(), actions.cpu()
        torque = torque.cpu()
        
        if self.tracking_Z_bool:
            labelsPlot = 'Z'
            index = -1
        else:
            labelsPlot = 'X'
            index = 0

        # Position Plot
        plt.figure(figsize=(size_1, size_2))
        plt.plot(timeTask[0:(self.progress_buf[0] + 1)], pos_tip[0:(self.progress_buf[0] + 1), index], linewidth=labelWidth, color='b', linestyle='solid', label=labelsPlot + ' Real')
        plt.plot(timeTask[0:(self.progress_buf[0] + 1)], pos_des[1, 0:self.progress_buf[0] + 1], linewidth=labelWidth, color='r', linestyle='solid', label=labelsPlot + ' Desired')
        plt.xlabel(r'$\mathbf{Time\,\, [s]}$', fontsize=font_size)
        plt.ylabel(r'$\mathbf{Position\,\, ' + labelsPlot + r'\,\, [m]}$', fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.grid()
        plt.tick_params(labelsize=font_size)
        plt.tight_layout()
        plt.savefig(name_folder + '/{}_{}_position.svg'.format(epoch, labelsPlot[index]), format='svg')
        plt.close()

        # Velocity Plot
        plt.figure(figsize=(size_1, size_2))
        plt.plot(timeTask[0:(self.progress_buf[0] + 1)], vel_tip[0:(self.progress_buf[0] + 1), index], linewidth=labelWidth, color='b', linestyle='solid', label=labelsPlot + ' Real')
        plt.plot(timeTask[0:(self.progress_buf[0] + 1)], vel_des[1, 0:self.progress_buf[0] + 1], linewidth=labelWidth, color='r', linestyle='solid', label=labelsPlot + ' Desired')
        plt.xlabel(r'$\mathbf{Time\,\, [s]}$', fontsize=font_size)
        plt.ylabel(r'$\mathbf{Velocity\,\, [m/s]}$', fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.grid()
        plt.tick_params(labelsize=font_size)
        plt.tight_layout()
        plt.savefig(name_folder + '/{}_{}_velocity.svg'.format(epoch, labelsPlot[index]), format='svg')
        plt.close()

        # Action Plot
        plt.figure(figsize=(size_1, size_2))
        plt.plot(timeTask[0:(self.progress_buf[0] + 1)], actions[0, 0:(self.progress_buf[0] + 1)], linewidth=labelWidth, color='r', linestyle='solid', label='Policy')
        plt.xlabel(r'$\mathbf{Time\,\, [s]}$', fontsize=font_size)
        plt.ylabel(r'$\mathbf{Action \,\, Position\,\, [rad]}$', fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.grid()
        plt.tick_params(labelsize=font_size)
        plt.tight_layout()
        plt.savefig(name_folder + '/{}_{}_action.svg'.format(epoch, labelsPlot), format='svg')
        plt.close()
        
        # Torque Plot
        plt.figure(figsize=(size_1, size_2))
        plt.plot(timeTask[0:(self.progress_buf[0] + 1)], torque[0, 0:(self.progress_buf[0] + 1)], linewidth=labelWidth, color='r', linestyle='solid', label='Policy')
        plt.xlabel(r'$\mathbf{Time\,\, [s]}$', fontsize=font_size)
        plt.ylabel(r'$\mathbf{Torque\,\, [Nm]}$', fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.grid()
        plt.tick_params(labelsize=font_size)
        plt.tight_layout()
        plt.savefig(name_folder + '/{}_{}_torque.svg'.format(epoch, labelsPlot), format='svg')
        plt.close()
