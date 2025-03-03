from pxr import Usd, Gf
import random
import os
from tqdm import tqdm
import argparse
import json
import math

# Morphological Parameter Ranges
TRUNK_MASS = (4.00, 28.00)
TRUNK_WIDTH = (0.09, 0.30)
TRUNK_HEIGHT = (0.11, 0.19)
TRUNK_LENGTH = (0.37, 0.65)

HIP_MASS = (0.30, 0.69)
HIP_LENGTH = (0.03, 0.05)

THIGH_MASS = (0.60, 4.00)
THIGH_WIDTH = (0.02, 0.04)
THIGH_HEIGHT = (0.03, 0.05)
THIGH_LENGTH = (0.21, 0.35)

CALF_MASS = (0.10, 0.86)
CALF_WIDTH = (0.016, 0.020)
CALF_HEIGHT = (0.013, 0.019)
CALF_LENGTH = (0.21, 0.35)

# Control Parameters Ranges
KP = (20, 80)
KD = (0.6, 2.0)
ALPHA = (0.9, 1.1)
DELTA_M = (-0.02, 0.02)

# Other Parameters
PAYLOAD = (-2.0, 2.0)
MOTOR_FRICTIONS = (0.2, 1.25) 

ETA_A = 0.03499
ETA_B = 60.3338
ETA_C = 1.382
ETA_D = -0.1001

def total_mass(stage: Usd.Stage):
    """
    Calculate the total mass of the robot.
    """
    total_mass = 0
    for prim in stage.Traverse():
        if prim.HasAttribute("physics:mass"):
            mass_attr = prim.GetAttribute("physics:mass")
            mass = mass_attr.Get()
            if mass is not None:
                total_mass += mass
    
    return total_mass

def round_params(params):
    if isinstance(params, tuple):
        rounded_params = []
        for param in params:
            rounded_params.append(round(param, 2))
        return rounded_params
    else:
        return round(params, 2)

def generate_quadrupeds(
    template_path: str,
    output_dir: str,
    num_envs: int,
    seed: int,
):
    random.seed(seed)
    # pre-generate all morphology parameters
    keys = [f"quadruped_{num}" for num in range(num_envs)]
    robot_params = {
        key: {
            'base': (random.uniform(*TRUNK_LENGTH), random.uniform(*TRUNK_WIDTH), random.uniform(*TRUNK_HEIGHT)),
            'base_mass': random.uniform(*TRUNK_MASS),
            'hip': random.uniform(*HIP_LENGTH),
            'hip_mass': random.uniform(*HIP_MASS),
            'thigh': (random.uniform(*THIGH_LENGTH),random.uniform(*THIGH_WIDTH),random.uniform(*THIGH_HEIGHT)),
            'thigh_mass': random.uniform(*THIGH_MASS),
            'calf': (random.uniform(*CALF_LENGTH),random.uniform(*CALF_WIDTH),random.uniform(*CALF_HEIGHT)),
            'calf_mass': random.uniform(*CALF_MASS),
            'motor_stiffness': random.uniform(*KP),
            'motor_damping': random.uniform(*KD)
        }
        for key in keys
    }

    print("Generating Morphologies...")
    # open quadruped template
    template_stage = Usd.Stage.Open(template_path)
    nomial_mass = float('inf')
    morphologies = {}
    # create new .usda files based on the generated parameters
    for key, value in tqdm(robot_params.items()):
        # create new stage and copy content from template
        new_stage = Usd.Stage.CreateNew(f"{output_dir}/{key}.usda")
        new_stage.GetRootLayer().TransferContent(template_stage.GetRootLayer())

        # update base
        base_prim = new_stage.GetPrimAtPath("/quadruped_template/base")
        base_length, base_width, base_height = round_params(value['base'])
        base_mass = round_params(value['base_mass'])
        
        base_prim.GetAttribute("xformOp:scale").Set((base_length, base_width, base_height))
        base_prim.GetAttribute("physics:mass").Set(base_mass)

        # update legs
        hip_radius = round_params(value['hip'])
        hip_mass = round_params(value['hip_mass'])

        thigh_length, thigh_width, thigh_height = round_params(value['thigh'])
        thigh_mass = round_params(value['thigh_mass'])

        calf_length, calf_width, calf_height = round_params(value['calf'])
        calf_mass = round_params(value['calf_mass'])

        legs = {
            "LF": (base_length/2, base_width/2+hip_radius, 0),
            "RF": (base_length/2, -base_width/2-hip_radius, 0),
            "LH": (-base_length/2, base_width/2+hip_radius, 0),
            "RH": (-base_length/2, -base_width/2-hip_radius, 0)
        }

        for leg, pos in legs.items():
            # update dimensions
            hip_prim = new_stage.GetPrimAtPath(f"/quadruped_template/{leg}_HIP")
            hip_prim.GetAttribute("radius").Set(hip_radius)
            hip_prim.GetAttribute("physics:mass").Set(hip_mass)

            thigh_prim = new_stage.GetPrimAtPath(f"/quadruped_template/{leg}_THIGH")
            thigh_prim.GetAttribute("xformOp:scale").Set((thigh_length, thigh_width, thigh_height))
            thigh_prim.GetAttribute("physics:mass").Set(thigh_mass)

            calf_prim = new_stage.GetPrimAtPath(f"/quadruped_template/{leg}_CALF")
            calf_prim.GetAttribute("xformOp:scale").Set((calf_length, calf_width, calf_height))
            calf_prim.GetAttribute("physics:mass").Set(calf_mass)

            foot_prim = new_stage.GetPrimAtPath(f"/quadruped_template/{leg}_FOOT")
            foot_joint_prim = new_stage.GetPrimAtPath(f"/quadruped_template/{leg}_CALF/{leg}_FF")

            # update positions
            if pos[1] > 0:
                hip_prim.GetAttribute("xformOp:translate").Set((pos[0], pos[1]-hip_radius, 0))
                thigh_prim.GetAttribute("xformOp:translate").Set((pos[0], pos[1]+thigh_width/2, -thigh_length/2))
                calf_prim.GetAttribute("xformOp:translate").Set(
                    (pos[0], pos[1]+thigh_width/2, -thigh_length-calf_length/2)
                )
                foot_prim.GetAttribute("xformOp:translate").Set(
                    (pos[0], pos[1]+thigh_width/2, -thigh_length-calf_length)
                )

            else:
                hip_prim.GetAttribute("xformOp:translate").Set((pos[0], pos[1]+hip_radius, 0))
                thigh_prim.GetAttribute("xformOp:translate").Set((pos[0], pos[1]-thigh_width/2, -thigh_length/2))
                calf_prim.GetAttribute("xformOp:translate").Set(
                    (pos[0], pos[1]-thigh_width/2, -thigh_length-calf_length/2)
                )
                foot_prim.GetAttribute("xformOp:translate").Set(
                    (pos[0], pos[1]-thigh_width/2, -thigh_length-calf_length)
                )
            # align fixed foot join
            foot_joint_prim.GetAttribute("physics:localPos0").Set((0, 0, calf_length/2))

            # update actuator gains
            stiffness = round_params(value['motor_stiffness'])
            damping = round_params(value['motor_damping'])
            
            haa_prim = new_stage.GetPrimAtPath(f"/quadruped_template/base/{leg}_HAA")
            haa_prim.GetAttribute("drive:angular:physics:stiffness").Set(stiffness)
            haa_prim.GetAttribute("drive:angular:physics:damping").Set(damping)

            hfe_prim = new_stage.GetPrimAtPath(f"/quadruped_template/{leg}_HIP/{leg}_HFE")
            hfe_prim.GetAttribute("drive:angular:physics:stiffness").Set(stiffness)
            hfe_prim.GetAttribute("drive:angular:physics:damping").Set(damping)

            kfe_prim = new_stage.GetPrimAtPath(f"/quadruped_template/{leg}_THIGH/{leg}_KFE")
            kfe_prim.GetAttribute("drive:angular:physics:stiffness").Set(stiffness)
            kfe_prim.GetAttribute("drive:angular:physics:damping").Set(damping)

        # find nomial mass and calulate mass ratio
        quadruped_mass = total_mass(new_stage)
        if quadruped_mass < nomial_mass:
            nomial_mass = quadruped_mass
        mass_ratio = quadruped_mass / nomial_mass
        
        # calculate gain scale
        eta = ETA_A * mass_ratio**3 + ETA_B * mass_ratio**2 + ETA_C * mass_ratio + ETA_D

        new_stage.GetRootLayer().Save()
        
        # store revelevant morphology params
        morphologies[key+'.usda'] = [
            base_length,
            base_width,
            base_height,
            base_mass,
            hip_mass,
            thigh_length,
            thigh_mass,
            calf_length,
            calf_mass,
            stiffness,
            damping,
            eta
        ]
        with open(os.path.join(output_dir, 'morphology_params.json'), 'w') as f:
            json.dump(morphologies, f)

if __name__ == "__main__":
    template_path = os.path.join(os.getcwd(), 'morphologies/quadruped_template.usda')
    output_dir = os.path.join(os.getcwd(), 'morphologies/generated_quads')
    generate_quadrupeds(
        template_path,
        output_dir,
        4096,
        42,
    )
