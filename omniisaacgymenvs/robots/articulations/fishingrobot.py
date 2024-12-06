from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
# from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
import carb
# from pxr import UsdGeom
# from pxr import PhysxSchema


class FishingRod(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "FishingRod",
        default_dof_pos: Optional[np.ndarray] = torch.zeros((22,)),
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None, 
        orientation: Optional[np.ndarray] = None, 
    ) -> None:

        self._usd_path = usd_path
        self._name = name
        self._default_dof_pos = default_dof_pos
        self._index_joint = range(0, len(self._default_dof_pos))
        

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            # self._usd_path = '/isaac-sim/OmniIsaacGymEnvs/omniisaacgymenvs/assets/robots/fishing_rod/fishing_rod.usd'
            self._usd_path = '/home/michele/Documents/OmniIsaacGymEnvs/omniisaacgymenvs/assets/robots/fishing_rod/fishing_rod.usd'
        add_reference_to_stage(self._usd_path, prim_path)
                
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        self._dof_names = ["fishing_actuator_joint",
                           "Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5",  
                           "Joint_6", "Joint_7", "Joint_8", "Joint_9", "Joint_10",  
                           "Joint_11", "Joint_12", "Joint_13", "Joint_14", "Joint_15", 
                           "Joint_16", "Joint_17", "Joint_18", "Joint_19", "Joint_20"]
        
        self._dof_names_actuated = ["fishing_actuator_joint"]
        

    @property
    def dof_names(self):
        return self._dof_names
    
    def set_to_default_configuration(self):
        default_dof_pos = torch.tensor(self._default_dof_pos, dtype=torch.float32, device=self._device, requires_grad=False)
        self.set_joint_positions(default_dof_pos)   
           
    

   