import numpy as np
from scipy.stats import norm
from typing import Tuple

from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from conditioning import get_cond_state
from solution import task2 as task2_solu
from scipy.stats import norm

# ex 2h


def get_conds(state: MultiVarGauss2d,
              sens_model_c: LinearSensorModel2d, meas_c: Measurement2d,
              sens_model_r: LinearSensorModel2d, meas_r: Measurement2d
              ) -> Tuple[MultiVarGauss2d, MultiVarGauss2d]:

    # TODO replace this with own code
    cond_c = get_cond_state(state, sens_model_c, meas_c)
    cond_r = get_cond_state(state, sens_model_r, meas_r)
    return cond_c, cond_r

# ex 2i


def get_double_conds(state: MultiVarGauss2d,
                     sens_model_c: LinearSensorModel2d, meas_c: Measurement2d,
                     sens_model_r: LinearSensorModel2d, meas_r: Measurement2d
                     ) -> Tuple[MultiVarGauss2d, MultiVarGauss2d]:

 # TODO
    cond_c, cond_r = get_conds(
        state, sens_model_c, meas_c, sens_model_r, meas_r)

    cond_cr = get_cond_state(cond_c, sens_model_r, meas_r)
    cond_rc = get_cond_state(cond_r, sens_model_c, meas_c)
    return cond_cr, cond_rc

# ex 2k


def get_prob_over_line(gauss: MultiVarGauss2d) -> float:

    # TODO replace this with own code
    # Pr(x_2-x_1>5)=
    prob = 1-norm.cdf(5, gauss.mean[1]-gauss.mean[0], np.sqrt(
        gauss.cov[0, 0]+gauss.cov[1, 1]-2*gauss.cov[0, 1]))[0]
    return prob
