import numpy as np
from senfuslib import MultiVarGauss, Simulator
from copy import deepcopy

from models import ModelCT, ModelCV, ModelImm
from states import StateCV
from sensors import SensorPos
from filter import FilterIMM
from gaussian_mixture import GaussianMixture

"""Simulation parameters"""
x0_sim = MultiVarGauss[StateCV](StateCV(0, 0, 6, 0),
                                np.diag([0.1, 0.1, 0.1, 0.1]))

dynamic_model_sim = ModelImm(
    models=[ModelCV(std_vel=1),
            ModelCT(std_vel=0.5, rate=0.3),
            ModelCT(std_vel=0.5, rate=-0.5)],
    hold_times=np.array([10, 6, 6]),
    jump_mat=np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]))

sensor_model_sim = SensorPos(std_pos=0.5)
sim_imm = Simulator(dynamic_model=dynamic_model_sim,
                    sensor_model=sensor_model_sim,
                    init_state=MultiVarGauss(
                        x0_sim.mean.with_new_meta(prev_mode=0),
                        x0_sim.cov),
                    dt=0.1,
                    end_time=80,
                    seed='None'
                    )


"""Filter parameters"""
x0_est = GaussianMixture(np.ones(3) / 3, [x0_sim, x0_sim, x0_sim])
sensor_model_filter = SensorPos(std_pos=.5)
dynamic_model_filter = deepcopy(dynamic_model_sim)
filter_imm = FilterIMM(dynamic_model_filter, sensor_model_filter)
