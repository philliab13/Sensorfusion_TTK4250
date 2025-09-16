from solution import ekf as ekf_solu
"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z are multiple measurements so that z = Z[k] at a given time step k
v is the innovation z - h(x)
S is the innovation covariance
"""
from dataclasses import dataclass
import numpy as np

from dynamicmodels import WhitenoiseAcceleration2D
from measurementmodels import CartesianPosition2D
from mytypes import MultiVarGauss, Measurement2d


@dataclass
class ExtendedKalmanFilter:
    dyn_modl: WhitenoiseAcceleration2D
    sens_modl: CartesianPosition2D

    def step(self,
             state_old: MultiVarGauss,
             meas: Measurement2d,
             ) -> MultiVarGauss:
        """Given previous state estimate and measurement, 
        return new state estimate.

        Relationship between variable names and equations in the book:
        \hat{x}_{k|k_1} = pres_state.mean
        P_{k|k_1} = pres_state.cov
        \hat{z}_{k|k-1} = pred_meas.mean
        \hat{S}_k = pred_meas.cov
        \hat{x}_k = upd_state_est.mean
        P_k = upd_state_est.cov
        """
        state_pred = self.dyn_modl.predict_state(state_old, meas.dt)
        meas_pred = self.sens_modl.predict_measurement(state_pred)
        z = meas.value

        x_est_mean = state_pred.mean  # 2
        P = state_pred.cov  # 3
        z_est_mean = meas_pred.mean  # 4
        innovation = z - z_est_mean  # 5

        H = self.sens_modl.H(x_est_mean)  # TODO

        kalman_gain = P@H.T@np.linalg.inv(meas_pred.cov)  # 7
        state_upd_mean = x_est_mean + kalman_gain@innovation  # 8
        state_upd_cov = P - kalman_gain@H@P  # 9

        state_upd = MultiVarGauss(state_upd_mean, state_upd_cov)

        # TODO replace this with own code

        return state_upd, state_pred, meas_pred
