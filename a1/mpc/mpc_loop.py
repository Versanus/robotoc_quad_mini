import os
import time
import numpy as np
import pybullet


def get_control_input(control_policy, q: np.ndarray, v: np.ndarray):
    """
    Computes torque using LQR policy (tau - Kp(q_err) - Kd(dq_err))
    """
    nJ = control_policy.tauJ.shape[0]
    qJ = q[-nJ:]
    dqJ = v[-nJ:]

    return (
        control_policy.tauJ
        - control_policy.Kp @ (control_policy.qJ - qJ)
        - control_policy.Kd @ (control_policy.dqJ - dqJ)
    )


class SimpleMPCSimulator:
    """
    Custom MPC loop:
        - PyBullet at high frequency (e.g., 1000 Hz)
        - MPC at lower frequency (e.g., 100 Hz)
    """

    def __init__(self, simulator, sim_dt=0.001, mpc_dt=0.01):
        self.simulator = simulator
        self.sim_dt = sim_dt
        self.mpc_dt = mpc_dt

        if mpc_dt < sim_dt:
            raise ValueError("mpc_dt must be >= sim_dt")

        self.steps_per_mpc = int(mpc_dt / sim_dt)

    def run(
        self,
        mpc,
        t0: float,
        q0: np.ndarray,
        simulation_time: float,
        log: bool = False,
        record: bool = False,
        name: str = "mpc_sim",
        print_every: int = 5,
    ):

        self.simulator.init_simulation(t0, q0)

        if record:
            pybullet.startStateLogging(
                pybullet.STATE_LOGGING_VIDEO_MP4, name + ".mp4"
            )

        # Logging setup
        if log:
            log_dir = os.path.join(os.getcwd(), name + "_log")
            os.makedirs(log_dir, exist_ok=True)

            q_log = open(os.path.join(log_dir, "q.log"), "w")
            v_log = open(os.path.join(log_dir, "v.log"), "w")
            u_log = open(os.path.join(log_dir, "u.log"), "w")
            t_log = open(os.path.join(log_dir, "t.log"), "w")
            kkt_log = open(os.path.join(log_dir, "kkt.log"), "w")

        step_counter = 0
        solve_counter = 0
        kkt_error = 0.0

        while self.simulator.get_time() < t0 + simulation_time:

            t = self.simulator.get_time()
            q, v = self.simulator.get_state()

            # ================================
            # MPC UPDATE (100 Hz typically)
            # ================================
            if step_counter % self.steps_per_mpc == 0:

                start = time.perf_counter()

                mpc.update_solution(t, self.mpc_dt, q, v)

                solve_time = time.perf_counter() - start
                kkt_error = mpc.KKT_error(t, q, v)

                solve_counter += 1

                if solve_counter % print_every == 0:
                    print(
                        f"[{solve_counter}] "
                        f"Solve: {solve_time*1000:.2f} ms | "
                        f"KKT: {kkt_error:.3e}"
                    )

            # ================================
            # CONTROL (1000 Hz)
            # ================================
            control_policy = mpc.get_control_policy(t)
            u = get_control_input(control_policy, q, v)

            self.simulator.step_simulation(u)

            # ================================
            # LOGGING
            # ================================
            if log:
                np.savetxt(q_log, [q])
                np.savetxt(v_log, [v])
                np.savetxt(u_log, [u])
                np.savetxt(t_log, [t])
                np.savetxt(kkt_log, [kkt_error])

            step_counter += 1

        if log:
            q_log.close()
            v_log.close()
            u_log.close()
            t_log.close()
            kkt_log.close()
            print(f"Logs saved to {log_dir}")

        if record:
            self.simulator.disconnect()