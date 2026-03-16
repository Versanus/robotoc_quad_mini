import robotoc
from robotoc_sim import MPCSimulation, CameraSettings, TerrainSettings
from a1_simulator import A1Simulator
import numpy as np
import mujoco
import mujoco.viewer
import time

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(THIS_DIR, "../../quad_mini_description/urdf/quad_mini_description_pybullet.urdf")
URDF_PATH = os.path.normpath(URDF_PATH)
XML_PATH = os.path.join(THIS_DIR, "../../quad_mini_description/quadmini_scenev2.xml")
XML_PATH = os.path.normpath(XML_PATH)

print("Using URDF for robotoc:", URDF_PATH)
print("Using XML for MuJoCo:", XML_PATH)
model_info = robotoc.RobotModelInfo()
model_info.urdf_path = URDF_PATH
model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
baumgarte_time_step = 0.05
model_info.point_contacts = [robotoc.ContactModelInfo('FL_FOOT', baumgarte_time_step),
                             robotoc.ContactModelInfo('FR_FOOT', baumgarte_time_step),
                             robotoc.ContactModelInfo('HL_FOOT', baumgarte_time_step),
                             robotoc.ContactModelInfo('HR_FOOT', baumgarte_time_step)]
robot = robotoc.Robot(model_info)

step_length = np.array([0.03, 0, 0])  # Moderate step length
step_yaw = 0.0

step_height = 0.04  # Moderate step height
swing_time = 0.25
stance_time = 0.05
swing_start_time = 0.50

vcom_cmd = 0.25 * step_length / (swing_time+stance_time)
yaw_rate_cmd = step_yaw / (swing_time+stance_time)

T = 0.5
N = 20
mpc = robotoc.MPCTrot(robot, T, N)

planner = robotoc.TrotFootStepPlanner(robot)
planner.set_gait_pattern(step_length, step_yaw, (stance_time > 0.))
# planner.set_raibert_gait_pattern(vcom_cmd, yaw_rate_cmd, swing_time, stance_time, gain=0.7)
mpc.set_gait_pattern(planner, step_height, swing_time, stance_time, swing_start_time)

t0 = 0.0
q0 = np.array([0, 0, 0.181, 0, 0, 0, 1,
               0, -0.17, 0.92,
               0, 0.17, 0.92,
               0, -0.17, 0.92,
               0, 0.17, 0.92])
#q0[0] -= 2.5
v0 = np.zeros(robot.dimv())
option_init = robotoc.SolverOptions()
option_init.max_iter = 10
option_init.nthreads = 4
mpc.init(t0, q0, v0, option_init)

option_mpc = robotoc.SolverOptions()
option_mpc.max_iter = 10  # MPC iterations (reduced for stability)
option_mpc.nthreads = 4
option_mpc.enable_benchmark = True  # Enable CPU time measurement
mpc.set_solver_options(option_mpc)

time_step = 0.0025 # 400 Hz control rate (20 steps per MPC solve at 20Hz)

# Initialize MuJoCo for visualization
print("\n=== Initializing MuJoCo Visualization ===")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print(f"MuJoCo model loaded. Number of bodies: {model.nbody}")
print(f"Number of joints: {model.njnt}")
print(f"Number of actuators: {model.na}")

# Get the base_link body ID
base_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
print(f"Base link body ID: {base_link_id}")

# Find joint and actuator indices
# The order should be: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, HL_hip, HL_thigh, HL_calf, HR_hip, HR_thigh, HR_calf
expected_joints = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "HL_hip_joint", "HL_thigh_joint", "HL_calf_joint",
    "HR_hip_joint", "HR_thigh_joint", "HR_calf_joint"
]

expected_actuators = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "HL_hip", "HL_thigh", "HL_calf",
    "HR_hip", "HR_thigh", "HR_calf"
]

joint_ids = []
actuator_ids = []
print(robot.dimu())
print(model.na)
for jnt_name in expected_joints:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    if jid >= 0:
        joint_ids.append(jid)

for act_name in expected_actuators:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    if act_id >= 0:
        actuator_ids.append(act_id)

print(f"Found {len(joint_ids)} controlled joints")
print(f"Found {len(actuator_ids)} actuators")
print("model.na =", model.na)
print("model.nu =", model.nu)
# Set initial state from q0
# Base position (x, y, z)
data.qpos[0:3] = q0[0:3]
# Base orientation (quaternion w, x, y, z)
data.qpos[3:7] = q0[3:7]
data.qpos[3]=0.0
data.qpos[6] = 1.0
# Set initial joint angles from q0
for idx, jid in enumerate(joint_ids[:12]):
    if idx < len(q0) - 7:
        qpos_addr = model.jnt_qposadr[jid]
        data.qpos[qpos_addr] = q0[7 + idx]

mujoco.mj_step(model, data)

print("\n=== Testing MPC Solver ===")
test_time = 0.0
solver = mpc.get_solver()
solver.solve(test_time, q0, v0)
print("\n=== MPC Solve Time Statistics ===")
stats = solver.get_solver_statistics()
print(stats)

# Run visualization with MuJoCo viewer
print("\n=== Running MuJoCo Visualization ===")
print("Close the viewer window to exit...")

simulation_time = 10.0
step_count = 0
current_sim_time = 0.0
q_current = q0.copy()
v_current = v0.copy()
u_current = np.zeros(robot.dimu())
last_solve_time = 0.0

# Create viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while current_sim_time < simulation_time and viewer.is_running():
        # Solve MPC at every step for maximum responsiveness 
        try:
            solve_start = time.time()
            
            # Update solution with current time and state
            mpc.update_solution(current_sim_time, time_step, q_current, v_current)
            
            last_solve_time = (time.time() - solve_start) * 1000
            
            # Get the control input from the MPC
            u_current = mpc.get_initial_control_input()
            u_current = np.asarray(u_current)
                
        except Exception as e:
            u_current = np.zeros(robot.dimu())
            print(f"MPC solve error at t={current_sim_time:.2f}: {e}")
        
        # Print diagnostics
        if step_count % 10 == 0:  # Every 0.25 seconds
            print(f"\n[t={current_sim_time:.2f}s] MPC Solve Time: {last_solve_time:.2f}ms")
            print(f"  State - q: {q_current[0:3]}, v_lin: {v_current[0:3]}")
            print(f"  DEBUG - q_current full: {q_current}")
            print(f"  DEBUG - v_current full: {v_current}")
            
            # Get reference trajectory for comparison
            solution = mpc.get_solution()
            if len(solution) > 0:
                ref_q = np.asarray(solution[0].q)
                print(f"  DEBUG - MPC ref q[0] pos: {ref_q[0:3]}")
            
            print(f"  Joint torques by leg (N⋅m):")
            for leg_idx, leg in enumerate(['FL', 'FR', 'HL', 'HR']):
                torques_leg = u_current[leg_idx*3:(leg_idx+1)*3]
                print(f"    {leg}: {torques_leg[0]:8.4f} {torques_leg[1]:8.4f} {torques_leg[2]:8.4f}")
            print(f"  Max torque: {np.max(np.abs(u_current)):.4f} N⋅m")
        
        # Apply control to actuators
        for idx, act_idx in enumerate(actuator_ids):
            if idx < len(u_current):
                torque = float(u_current[idx])
                data.ctrl[act_idx] = torque
        
        # Step MuJoCo simulation
        mujoco.mj_step(model, data)
        
        # Update state from MuJoCo
        q_current[0:3] = data.qpos[0:3]
        # MuJoCo gives [w, x, y, z]
        qw = data.qpos[3]
        qx = data.qpos[4]
        qy = data.qpos[5]
        qz = data.qpos[6]

        # Convert to robotoc format [x, y, z, w]
        q_current[3] = qx
        q_current[4] = qy
        q_current[5] = qz
        q_current[6] = qw

        # Normalize quaternion
        q_current[3:7] /= np.linalg.norm(q_current[3:7])
        for idx, jid in enumerate(joint_ids[:12]):
            jnt_addr = model.jnt_qposadr[jid]
            q_current[7 + idx] = data.qpos[jnt_addr]
        
        v_current[0:3] = data.qvel[0:3]
        v_current[3:6] = data.qvel[3:6]
        for idx, jid in enumerate(joint_ids[:12]):
            jnt_vel_addr = model.jnt_dofadr[jid]
            v_current[6 + idx] = data.qvel[jnt_vel_addr]
        
        # Update viewer
        viewer.sync()
        time.sleep(time_step)
        current_sim_time += time_step
        step_count += 1

print("Visualization complete!")

print("PyBullet disconnected.")


