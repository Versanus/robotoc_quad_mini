import matplotlib

import robotoc
from quad_simulator import QuadSimulator
from robotoc_sim import MPCSimulation, CameraSettings, TerrainSettings
import numpy as np
import os
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpc_loop import SimpleMPCSimulator

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(THIS_DIR, "../../quad_mini_description/urdf/quad_mini_description_pybullet.urdf")
URDF_PATH = os.path.normpath(URDF_PATH)

print("Using URDF:", URDF_PATH)


model_info = robotoc.RobotModelInfo()
model_info.urdf_path = URDF_PATH
model_info.base_joint_type = robotoc.BaseJointType.FloatingBase
baumgarte_time_step = 0.05
model_info.point_contacts = [robotoc.ContactModelInfo('FL_FOOT', baumgarte_time_step),
                             robotoc.ContactModelInfo('HL_FOOT', baumgarte_time_step),
                             robotoc.ContactModelInfo('FR_FOOT', baumgarte_time_step),
                             robotoc.ContactModelInfo('HR_FOOT', baumgarte_time_step)]
robot = robotoc.Robot(model_info)

step_length = np.array([-0.08, 0, 0]) 
# step_length = np.array([-0.1, 0, 0]) 
# step_length = np.array([0, 0.1, 0]) 
# step_length = np.array([0.1, -0.1, 0]) 
step_yaw = 0.1
# step_yaw = 0.0

step_height = 0.06
swing_time = 0.25
stance_time = 0.05
# stance_time = 0.05
swing_start_time = 0.30

vcom_cmd = 0.5 * step_length / (swing_time+stance_time)
yaw_rate_cmd = step_yaw / (swing_time+stance_time)
option_mpc = robotoc.SolverOptions()
option_mpc.nthreads = 4

T = 0.4 # total time horizon
N = 20 # number of discretization steps

option_mpc.max_iter = 2 # MPC iterations


mpc = robotoc.MPCTrot(robot, T, N)

planner = robotoc.TrotFootStepPlanner(robot)
planner.set_gait_pattern(step_length, step_yaw, (stance_time>0.))
# planner.set_raibert_gait_pattern(vcom_cmd, yaw_rate_cmd, swing_time, stance_time, gain=0.7)
mpc.set_gait_pattern(planner, step_height, swing_time, stance_time, swing_start_time)

t0 = 0.0
q0 = np.array([
        0, 0, 0.200-0.002,
        0, 0, 0, 1,
        0.0, -0.25, 0.99,
        0.0,  0.25, 0.99,
        0.0, -0.25, 0.99,
        0.0,  0.25, 0.99
    ])
v0 = np.zeros(robot.dimv())
option_init = robotoc.SolverOptions()
option_init.max_iter = 50
option_init.nthreads = 4

# ================================
# Restore ALL default MPCTrot weights
# ================================

config_cost = mpc.get_config_cost_handle()
base_rot_cost = mpc.get_base_rotation_cost_handle()
swing_costs = mpc.get_swing_foot_cost_handle()
com_cost = mpc.get_com_cost_handle()

# ---- Configuration Cost ----

# q weight
q_weight = np.ones(robot.dimv()) * 0.005  #0.001
q_weight[:6] = 0.0
config_cost.set_q_weight(q_weight)
config_cost.set_q_weight_terminal(q_weight)

# impact q weight
q_weight_impact = np.ones(robot.dimv()) * 1.00 #1.0
q_weight_impact[:6] = 0.0
config_cost.set_q_weight_impact(q_weight_impact)

# velocity weights
v_weight = np.ones(robot.dimv()) * 1.00 #1.0
config_cost.set_v_weight(v_weight)
config_cost.set_v_weight_terminal(v_weight)

# impact velocity
config_cost.set_v_weight_impact(np.ones(robot.dimv()) * 1.00) #1.0

# impact dv weight
config_cost.set_dv_weight_impact(np.ones(robot.dimv()) *1e-3) #1e-3

# torque weight
config_cost.set_u_weight(np.ones(robot.dimu()) * 0.2e-2) #0.01


# ---- Base Rotation Cost ----

base_rot_weight = np.zeros(robot.dimv())
base_rot_weight[3:6] =50.0 # base_rot_weight[3:6] = 1000.0
base_rot_cost.set_q_weight(base_rot_weight)
base_rot_cost.set_q_weight_terminal(base_rot_weight)
base_rot_cost.set_q_weight_impact(base_rot_weight)


# ---- Swing Foot Costs ----

for foot_cost in swing_costs:
    foot_cost.set_weight(np.ones(3) * 2e4) # foot_cost.set_weight(np.ones(3) * 1e4)


# ---- CoM Cost ----

com_cost.set_weight(np.ones(3) * 16e3) # com_cost.set_weight(np.ones(3) * 1e3)

# ================================
print("Default MPCTrot weights restored.")
# ================================

mpc.init(t0, q0, v0, option_init)


mpc.set_solver_options(option_mpc)

sim_time_step = 0.002 # 1000 Hz
mpc_time_step = 0.01 # 100 Hz

simulator = QuadSimulator(
    urdf_path=model_info.urdf_path,
    time_step=sim_time_step  # 1000 Hz
)

mpc_runner = SimpleMPCSimulator(
    simulator,
    sim_dt=sim_time_step,   # 1000 Hz
    mpc_dt=mpc_time_step     # 100 Hz
)
terrain_settings = TerrainSettings(from_urdf=True)
#simulator.set_terrain_settings(terrain_settings)

simulation_time = 50.0
log = True
record = False
mpc_runner.run(
    mpc=mpc,
    t0=t0,
    q0=q0,
    simulation_time=simulation_time,
    log=log,
    record=record,
    name="a1_trot_terrain"
)

if log:
    import numpy as np
    import matplotlib.pyplot as plt

    log_dir = "a1_trot_log"

    t_log = np.genfromtxt(os.path.join(log_dir, "t.log"))
    u_log = np.genfromtxt(os.path.join(log_dir, "u.log"))
    v_log = np.genfromtxt(os.path.join(log_dir, "v.log"))

    # ---- Front Left Torques ----
    tau_FL = u_log[:, 0:3]

    plt.figure()
    plt.plot(t_log, tau_FL[:, 0], label="Hip")
    plt.plot(t_log, tau_FL[:, 1], label="Thigh")
    plt.plot(t_log, tau_FL[:, 2], label="Calf")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title("Front Left Leg Torques")
    plt.legend()
    plt.grid(True)
    plt.savefig("front_left_torques.png", dpi=300)
    print("Saved: front_left_torques.png")

    # ---- Knee (Calf) Torque + Velocity in RPM ----
    knee_tau = u_log[:, 2]        # torque [N·m]
    knee_vel = v_log[:, 8]        # velocity [rad/s]

    # Convert to RPM
    knee_rpm = knee_vel * 60.0 / (2.0 * np.pi)

    plt.figure()
    plt.plot(t_log, knee_tau, label="Knee Torque [N·m]")
    plt.plot(t_log, knee_rpm, label="Knee Velocity [RPM]")

    plt.xlabel("Time [s]")
    plt.title("Front Left Knee Torque & Velocity")
    plt.legend()
    plt.grid(True)
    plt.ylim(-25, 25)   # <-- LIMIT RANGE   
    plt.savefig("front_left_knee_tau_rpm.png", dpi=300)
    print("Saved: front_left_knee_tau_rpm.png")