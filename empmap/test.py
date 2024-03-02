import numpy as np
from empmap.emp_setup import MapSetup


def test_code():
    newmap = MapSetup(100, "type O", 3, 8.0, calc_dir='../newmap/')
    newmap.load_universe(topology="../traj_ex/step5_1.gro",
                         trajectory="../traj_ex/step5_1.xtc")
    newmap.grab_clusters_from_frames([0, 1, 2])
    return


if __name__ == "__main__":
    test_code()
