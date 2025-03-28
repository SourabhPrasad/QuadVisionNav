import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

ASSET_PATH = os.path.join(os.path.abspath(os.getcwd()), "morphologies")
print(ASSET_PATH)

SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    effort_limit=23.5,
    saturation_effort=23.5,
    velocity_limit=30.0,
    stiffness={".*": 20.0},
    damping={".*": 0.6},
)
"""Configuration for simple DC actuator for  use with custom generated quadrupeds."""

QUAD_TEMPLATE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSET_PATH, 'quadruped_87.usda'),
        activate_contact_sensors=True,          
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*LF_HAA": 0.2,  
            ".*RF_HAA": -0.2,
            ".*LH_HAA": 0.2,
            ".*RH_HAA": -0.2,
            ".*F_HFE": -0.4,  # both front HFE
            ".*H_HFE": 0.4,  # both hind HFE
            ".*F_KFE": 0.8,  # both front KFE
            ".*H_KFE": -0.8,  # both hind KFE
        },
    ),
    actuators={"legs": SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of quadruped using quadruped_template.usda"""

QUAD_TEMPLATE_CYL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSET_PATH, 'quadruped_template_cyl.usda'),
        activate_contact_sensors=True,          
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.2,            
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=None,
            damping=None,
            friction=0.0,
        ),
    },
)
"""Configuration of quadruped using quadruped_template_cyl.usda"""