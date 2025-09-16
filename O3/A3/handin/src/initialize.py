import numpy as np

from mytypes import Measurement2d, MultiVarGauss
from tuning import EKFParams
from solution import initialize as initialize_solu


def get_init_CV_state(meas0: Measurement2d, meas1: Measurement2d,
                      ekf_params: EKFParams) -> MultiVarGauss:
    """This function will estimate the initial state and covariance from
    the two first measurements"""
    dt = meas1.dt
    z0, z1 = meas0.value, meas1.value
    sigma_a = ekf_params.sigma_a
    sigma_z = ekf_params.sigma_z

    I2 = np.eye(2)

    H = np.hstack([I2, np.zeros((2, 2))])

    F = np.block([[I2, dt*I2],
                  [np.zeros((2, 2)), I2]])

    F_inv = np.block([[I2, -dt*I2],
                      [np.zeros((2, 2)), I2]])

    Q = np.block([
        [((dt**3)/3)*(sigma_a**2)*I2, ((dt**2)/2)*(sigma_a**2)*I2],
        [((dt**2)/2)*(sigma_a**2)*I2, (dt)*(sigma_a**2)*I2]
    ])

    top_left = (sigma_z**2) * I2
    top_right = (sigma_z**2 / dt) * I2
    bottom_left = top_right
    bottom_right = (2/(dt**2)) * (sigma_z**2) * I2 + \
        (1/(dt**2))*(H@F_inv@Q@(H@F_inv).T)

    cov = np.block([
        [top_left,    top_right],
        [bottom_left, bottom_right]
    ])

    vx = (z1[0] - z0[0]) / dt
    vy = (z1[1] - z0[1]) / dt
    mean = np.array([z1[0], z1[1], vx, vy])

    init_state = MultiVarGauss(mean, cov)

    return init_state
