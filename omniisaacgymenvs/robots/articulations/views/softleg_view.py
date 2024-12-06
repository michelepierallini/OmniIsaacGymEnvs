from typing import Optional
import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView, XFormPrimView

class JumpingSoftlegView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "JumpingSoftlegView",
        track_contact_forces=True, 
        prepare_contact_sensors=True,
        default_dof_pos: Optional[np.ndarray] = [0.11, 0.75, -0.75]
    ) -> None:
        """[summary]
        """
        self._default_dof_pos = default_dof_pos 
        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )
        self._foot = RigidPrimView(prim_paths_expr="/World/envs/.*/softleg_cart/softleg_1_calf_link",  
                                name="foot_view", 
                                reset_xform_properties=False, 
                                track_contact_forces=track_contact_forces, 
                                prepare_contact_sensors=prepare_contact_sensors)
        
        # self._foot_real = XFormPrimView(prim_paths_expr="/World/envs/.*/softleg_cart/softleg_1_calf_link/foot_link", 
        #                         name="foot_real_view", 
        #                         reset_xform_properties=False)
        
        self._cart = RigidPrimView(prim_paths_expr="/World/envs/.*/softleg_cart/softleg_1_cart_link",
                                name="cart_view", 
                                reset_xform_properties=False)
        
    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        

    def is_foot_not_toching_ground(self, threshold=1e-2):
        foot_pos, _ = self._foot.get_local_poses()
        foot_height = foot_pos[:, 2]
        return (foot_height[:] < threshold), foot_height
    
    def cart_position(self):
        cart_pos, _ = self._cart.get_local_poses()
        return cart_pos

    def set_to_default_configuration(self):
        default_dof_pos = np.array(self._default_dof_pos, dtype=np.float32)
        default_dof_pos = torch.tensor(default_dof_pos, dtype=torch.float32, device=self._device, requires_grad=False)
        self.set_joint_positions(default_dof_pos)
    
    def get_contact_forces(self):
        foot_forces = self._foot.get_net_contact_forces()
        return foot_forces    # (num_envs, 1, 3 dimensions force)
    
    
