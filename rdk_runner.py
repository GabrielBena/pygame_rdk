# random dot motion kinematogram
# timo flesch, 2019
# import matplotlib.pyplot as plt
# from array2gif import write_gif

import rdktools.rdk_params as params
from rdktools.rdk_experiment import Experiment
import numpy as np


def main():
    exp = Experiment()
    exp.run()
    print(f"Avg Error : {np.mean(exp.results[:, 1])}")


if __name__ == "__main__":
    main()
