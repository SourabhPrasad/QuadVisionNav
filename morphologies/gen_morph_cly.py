"""
Morphology Randomization

Based on a template usd file, generate a bunch of new usd files with random morphologies.
"""
from pxr import Usd, Gf
import random
import os
from tqdm import tqdm
import argparse
import json
import math

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

def euler_to_quat(pitch, yaw, roll):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return Gf.Quatf(w, x, y, z)

def randomize_quadruped(
        template_path: str,
        target_dir: str,
        num_envs: int,
        seed: int,
        size_factor_range: tuple[float, float],
        base_length_range: tuple[float, float],
        base_width_range: tuple[float, float],
        base_height_range: tuple[float, float],
        base_mass_range: tuple[float, float],
        thigh_length_range: tuple[float, float],
        thigh_radius_range: tuple[float, float],
        thigh_mass_range: tuple[float, float],
        calf_length_range: tuple[float, float],
        calf_radius_range: tuple[float, float],
        calf_mass_range: tuple[float, float],
        motor_p_gain_range: tuple[float, float],
        motor_d_gain_range: tuple[float, float],
        center_of_mass_range: tuple[float, float] | None = None,
):
    """
    Randomize the morphology of the quadruped and save to new USD files.
    """
    # set random seed
    random.seed(seed)

    # Load the template stage once
    template_stage = Usd.Stage.Open(template_path)

    # Get the total mass of the template robot
    template_total_mass = total_mass(template_stage)
    # Get the PD gains of the template robot
    template_motor_p_gain = template_stage.GetPrimAtPath("/quadruped/FL_hip/FL_thigh_joint").GetAttribute("drive:angular:physics:stiffness").Get()
    template_motor_d_gain = template_stage.GetPrimAtPath("/quadruped/FL_hip/FL_thigh_joint").GetAttribute("drive:angular:physics:damping").Get()

    # Pre-generate all random values
    robot_values = [
        {
            'size_factor': random.uniform(*size_factor_range),
            'base': (random.uniform(*base_length_range), random.uniform(*base_width_range), random.uniform(*base_height_range)),
            'base_mass': random.uniform(*base_mass_range),
            'thigh': (random.uniform(*thigh_length_range), random.uniform(*thigh_radius_range)),
            'thigh_mass': random.uniform(*thigh_mass_range),
            'calf': (random.uniform(*calf_length_range), random.uniform(*calf_radius_range)),
            'calf_mass': random.uniform(*calf_mass_range),
            'motor_p_gain': random.uniform(*motor_p_gain_range),
            'motor_d_gain': random.uniform(*motor_d_gain_range),
            # 'center_of_mass': random.uniform(*center_of_mass_range),
        }
        for _ in range(num_envs)
    ]
    min_base_volume = size_factor_range[0]**3 * base_length_range[0] * base_width_range[0] * base_height_range[0]
    max_base_volume = size_factor_range[1]**3 * base_length_range[1] * base_width_range[1] * base_height_range[1]

    # elif isinstance(size_factor_range, float):
    #     # no randomization
    #     print("\nNo randomization applied because input is not a range, but a value!\n")
    #     robot_values = [
    #         {
    #             'size_factor': size_factor_range,
    #             'base': (base_length_range, base_width_range, base_height_range),
    #             'base_mass': base_mass_range,
    #             'thigh': (thigh_length_range, thigh_radius_range),
    #             'thigh_mass': thigh_mass_range,
    #             'calf': (calf_length_range, calf_radius_range),
    #             'calf_mass': calf_mass_range,
    #             'motor_p_gain': motor_p_gain_range,
    #             'motor_d_gain': motor_d_gain_range,
    #             # 'center_of_mass': center_of_mass_range,
    #         }
    #         for _ in range(num_envs)
    #     ]
    # else:
    #     raise ValueError("Invalid input range for size_factor_range")
        
    for i, values in enumerate(tqdm(robot_values)):
        new_stage = Usd.Stage.CreateNew(f"{target_dir}/quadruped_modified_{i}.usda")
        new_stage.GetRootLayer().TransferContent(template_stage.GetRootLayer())

        size_factor = values['size_factor']

        # Randomize base
        base_prim = new_stage.GetPrimAtPath("/quadruped/base")
        base_length, base_width, base_height = [v * size_factor for v in values['base']]
        base_volume = base_length * base_width * base_height
        base_mass_mean = (base_volume-min_base_volume) / (max_base_volume-min_base_volume) * (base_mass_range[1]-base_mass_range[0]) + base_mass_range[0]
        base_mass_stddev = (base_mass_range[1]-base_mass_range[0]) / 12
        base_mass = random.gauss(base_mass_mean, base_mass_stddev)
        
        # base_mass = size_factor * values['base_mass']
        base_prim.GetAttribute("xformOp:scale").Set((base_length, base_width, base_height))
        base_prim.GetAttribute("physics:mass").Set(base_mass)

        # TODO Adjust center of mass
        # com_prim = new_stage.GetPrimAtPath("/quadruped/base/center_of_mass")
        # com_z = values['center_of_mass'] * base_height
        # com_prim.GetAttribute("xformOp:translate").Set((0, 0, com_z))

        # Randomize legs
        thigh_length, thigh_radius = [v * size_factor for v in values['thigh']]
        thigh_mass = size_factor * values['thigh_mass']
        calf_length, calf_radius = [v * size_factor for v in values['calf']]
        calf_mass = size_factor * values['calf_mass']
        # hip_radius = 0.06
        hip_radius = new_stage.GetPrimAtPath("/quadruped/FL_hip").GetAttribute("radius").Get()
        foot_radius = new_stage.GetPrimAtPath("/quadruped/FL_foot").GetAttribute("radius").Get()
        # Randomize legs, align links and joints

        # get the total mass of the new robot
        hip_mass = new_stage.GetPrimAtPath("/quadruped/FL_hip").GetAttribute("physics:mass").Get()
        foot_mass = new_stage.GetPrimAtPath("/quadruped/FL_foot").GetAttribute("physics:mass").Get()
        new_total_mass = base_mass + 4 * thigh_mass + 4 * calf_mass + 4 * hip_mass + 4 * foot_mass

        # compute PD gains, according to the GenLoco paper
        motor_p_gain = template_motor_p_gain * new_total_mass / template_total_mass * values['motor_p_gain']
        motor_d_gain = template_motor_d_gain * new_total_mass / template_total_mass * values['motor_d_gain']
        # motor_p_gain = (template_motor_p_gain + 0.375*(new_total_mass-template_total_mass)) * values['motor_p_gain']
        # motor_d_gain = (template_motor_d_gain + 0.1125*(new_total_mass-template_total_mass)) * values['motor_d_gain']

        # 0.8 can be removed if we want to align hip and body edge
        leg_positions = {
            "FL": (base_length/2+hip_radius, base_width/2-hip_radius*0.2, 0),
            "FR": (base_length/2+hip_radius, -base_width/2+hip_radius*0.2, 0),
            "RL": (-base_length/2-hip_radius, base_width/2-hip_radius*0.2, 0),
            "RR": (-base_length/2-hip_radius, -base_width/2+hip_radius*0.2, 0)
        }
        for leg, leg_pos in leg_positions.items():
            # Set hip position
            hip_prim = new_stage.GetPrimAtPath(f"/quadruped/{leg}_hip")
            # hip_prim.GetAttribute("radius").Set(hip_radius)
            hip_prim.GetAttribute("xformOp:translate").Set(leg_pos)

            # Set thigh size, position and mass
            thigh_prim = new_stage.GetPrimAtPath(f"/quadruped/{leg}_thigh")
            thigh_prim.GetAttribute("height").Set(thigh_length)
            thigh_prim.GetAttribute("radius").Set(thigh_radius)
            thigh_prim.GetAttribute("xformOp:translate").Set((leg_pos[0], leg_pos[1], -thigh_length/2))
            thigh_prim.GetAttribute("physics:mass").Set(thigh_mass)

            # Set calf size, position and mass
            calf_prim = new_stage.GetPrimAtPath(f"/quadruped/{leg}_calf")
            calf_prim.GetAttribute("height").Set(calf_length)
            calf_prim.GetAttribute("radius").Set(calf_radius)
            calf_prim.GetAttribute("xformOp:translate").Set((leg_pos[0], leg_pos[1], -thigh_length-calf_length/2))
            calf_prim.GetAttribute("physics:mass").Set(calf_mass)

            # Set foot position
            foot_prim = new_stage.GetPrimAtPath(f"/quadruped/{leg}_foot")
            foot_prim.GetAttribute("xformOp:translate").Set((leg_pos[0], leg_pos[1], -thigh_length - calf_length - foot_radius*0.5))

            # Align joints
            # Note that for hip joints, all pos needs to be divided by base_length and base_width because they are scaled
            hip_joint_prim = new_stage.GetPrimAtPath(f"/quadruped/base/{leg}_hip_joint")
            hip_joint_prim.GetAttribute("physics:localPos0").Set((leg_pos[0]/base_length, leg_pos[1]/base_width, 0.0))
            hip_joint_prim.GetAttribute("drive:angular:physics:stiffness").Set(motor_p_gain)
            hip_joint_prim.GetAttribute("drive:angular:physics:damping").Set(motor_d_gain)

            thigh_joint_prim = new_stage.GetPrimAtPath(f"/quadruped/{leg}_hip/{leg}_thigh_joint")
            thigh_joint_prim.GetAttribute("physics:localPos1").Set((0, 0, thigh_length/2))
            thigh_joint_prim.GetAttribute("drive:angular:physics:stiffness").Set(motor_p_gain)
            thigh_joint_prim.GetAttribute("drive:angular:physics:damping").Set(motor_d_gain)

            calf_joint_prim = new_stage.GetPrimAtPath(f"/quadruped/{leg}_thigh/{leg}_calf_joint")
            calf_joint_prim.GetAttribute("physics:localPos0").Set((0, 0, -thigh_length/2))
            calf_joint_prim.GetAttribute("physics:localPos1").Set((0, 0, calf_length/2))
            calf_joint_prim.GetAttribute("drive:angular:physics:stiffness").Set(motor_p_gain)
            calf_joint_prim.GetAttribute("drive:angular:physics:damping").Set(motor_d_gain)

            foot_joint_prim = new_stage.GetPrimAtPath(f"/quadruped/{leg}_calf/{leg}_foot_joint")
            foot_joint_prim.GetAttribute("physics:localPos0").Set((0, 0, -calf_length/2-foot_radius*0.5))

        new_stage.GetRootLayer().Save()

        # update the parameters with actual values
        values['base'] = (base_length, base_width, base_height)
        values['base_mass'] = base_mass
        values['thigh'] = (thigh_length, thigh_radius)
        values['thigh_mass'] = thigh_mass
        values['calf'] = (calf_length, calf_radius)
        values['calf_mass'] = calf_mass
        values['motor_p_gain'] = motor_p_gain
        values['motor_d_gain'] = motor_d_gain

    # store the parameters
    with open(os.path.join(target_dir, "quadruped_params.json"), "w") as f:
        json.dump(robot_values, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomize robot base morphology")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir_name", type=str, help="name of the output folder, can be small_quad, med_quad, etc.")
    args = parser.parse_args()

    seed = args.seed
    num_envs = args.num_envs

    # get the root directory (path_to_x_nav)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    output_dir = os.path.join(root_dir, f"morphologies/{args.output_dir_name}")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    template_path=os.path.join(root_dir, "morphologies/quadruped_template_cyl.usda")
    print(f"Generating {num_envs} quadrupeds")
    print(f"Using template path: {template_path}")
    print(f"Random seed: {seed}")
    randomize_quadruped(
        template_path=template_path,
        target_dir=output_dir,
        num_envs=num_envs,
        seed=seed,
        size_factor_range=(0.8, 1.3),   # small (0.9, 1.1)
        base_length_range=(0.3, 0.7),   # small (0.3, 0.7)
        base_width_range=(0.2, 0.3),    # small (0.2, 0.5)
        base_height_range=(0.08, 0.16),  # small (0.08, 0.2)
        base_mass_range=(6, 15),         # small (6, 8)
        thigh_length_range=(0.2, 0.35),
        thigh_radius_range=(0.02, 0.035),
        thigh_mass_range=(0.7, 1.3),
        calf_length_range=(0.15, 0.3),
        calf_radius_range=(0.02, 0.035),
        calf_mass_range=(0.1, 0.2),
        motor_p_gain_range=(0.7, 1.3),  # small (0.7, 1.3)
        motor_d_gain_range=(0.7, 1.3),  # small (0.7, 1.3)
    )