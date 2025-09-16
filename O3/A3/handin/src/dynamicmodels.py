from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from mytypes import MultiVarGauss
from solution import dynamicmodels as dynamicmodels_solu


@dataclass
class WhitenoiseAcceleration2D:
    """
    A white noise acceleration model, also known as constan velocity.
    States are position and speed.
    """
    sigma_a: float  # noise standard deviation

    def f(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the zero noise transition from x given dt."""

        x_next = self.F(x, dt)@x

        return x_next

    def F(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the discrete transition matrix given dt
        See (4.64) in the book."""

        F = np.block([[np.eye(2), dt*np.eye(2)],
                     [np.zeros((2, 2)), np.eye(2)]])
        return F

    def Q(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the discrete transition Covariance.
        See(4.64) in the book."""
        I2 = np.eye(2)

        Q = np.block([
            [((dt**3)/3)*(self.sigma_a**2)*I2, ((dt**2)/2)*(self.sigma_a**2)*I2],
            [((dt**2)/2)*(self.sigma_a**2)*I2, (dt)*(self.sigma_a**2)*I2]
        ])
        return Q

    def predict_state(self,
                      state_est: MultiVarGauss,
                      dt: float,
                      ) -> MultiVarGauss:
        """Given the current state estimate, 
        calculate the predicted state estimate.
        See 2. and 3. of Algorithm 1 in the book."""
        x_upd_prev, P = state_est

        F = self.F(x_upd_prev, dt)  # TODO
        Q = self.Q(x_upd_prev, dt)  # TODO

        x_pred = F @ x_upd_prev  # TODO
        P_pred = F@P@F.T + Q  # TODO

        state_pred_gauss = MultiVarGauss(x_pred, P_pred)

        return state_pred_gauss
