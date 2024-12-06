from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
import carb
# from pxr import UsdGeom
from pxr import PhysxSchema


class Softleg(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Softleg",
        default_dof_pos: Optional[np.ndarray] = [0.11, 2.494075501099897, -3.244567079457459],
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None, # [torch.tensor]
        orientation: Optional[np.ndarray] = None, # [torch.tensor]
    ) -> None:

        self._usd_path = usd_path
        self._name = name
        self._default_dof_pos = default_dof_pos
        self._index_joint = range(0, len(self._default_dof_pos))

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            # self._usd_path = '/isaac-sim/OmniIsaacGymEnvs/omniisaacgymenvs/assets/robots/softleg_description/softleg-isaac.usd'
            self._usd_path = '/home/michele/Documents/OmniIsaacGymEnvs/omniisaacgymenvs/assets/robots/softleg_description/softleg-isaac.usd'

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        self._dof_names = ["softleg_1_cart_joint",
                           "softleg_1_hip_joint", 
                           "softleg_1_knee_joint"]
        
        self._dof_names_actuated = ["softleg_1_hip_joint", 
                                    "softleg_1_knee_joint"]
        

    @property
    def dof_names(self):
        return self._dof_names
    
    def set_to_default_configuration(self):
        ## questa non funziona a questo livello 
        default_dof_pos = np.array(self._default_dof_pos, dtype=np.float32)
        default_dof_pos = torch.tensor(default_dof_pos, dtype=torch.float32, device=self._device, requires_grad=False)
        self.set_joint_positions(default_dof_pos)   
           
    def set_softleg_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64 / np.pi * 180)

    def prepare_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                components_wo_cr = ['softleg_1_cart_joint', 
                                    'softleg_1_hip_joint',
                                    'softleg_1_knee_joint'] # ,'softleg_1_contact_link']
                condition = not any([link_type in str(link_prim.GetPrimPath()) for link_type in components_wo_cr])
                if condition:
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)