from typing import Optional
import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

class FishingRodView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "FishingRodView",
        track_contact_forces=False, 
        prepare_contact_sensors=False,
        default_dof_pos: Optional[np.ndarray] = torch.zeros((22, ))
    ) -> None:
        """[summary]
        """
        self._default_dof_pos = default_dof_pos 
        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )        

        self._tip = RigidPrimView(prim_paths_expr="/World/envs/.*/fishingrod/Tip",  
                                name="tip_view", 
                                reset_xform_properties=False, 
                                track_contact_forces=track_contact_forces, 
                                prepare_contact_sensors=prepare_contact_sensors)
        
        self._base = RigidPrimView(prim_paths_expr="/World/envs/.*/fishingrod/fishing_acutator_link",  
                                name="base_view", 
                                reset_xform_properties=False, 
                                track_contact_forces=track_contact_forces, 
                                prepare_contact_sensors=prepare_contact_sensors)
        
        
    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
    
   
    def set_to_default_configuration(self):
        self.set_joint_positions(self._default_dof_pos)
    
    
    
