import robotoc_sim


class QuadSimulator(robotoc_sim.LeggedSimulator):
    def __init__(self, urdf_path, time_step):
        super().__init__(urdf_path, time_step)

    @classmethod
    def get_joint_id_map(self):
        return [
            1, 2, 3,      # FL
            5, 6, 7,      # FR
            9, 10, 11,    # HL
            13, 14, 15    # HR
        ]