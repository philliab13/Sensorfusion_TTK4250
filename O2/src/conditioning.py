import numpy as np
from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from solution import conditioning as conditioning_solu

# ex 2g


def get_cond_state(state: MultiVarGauss2d,
                   sens_modl: LinearSensorModel2d,
                   meas: Measurement2d
                   ) -> MultiVarGauss2d:
    # Formulas from Alg 1.  p56
    pred_meas = sens_modl.get_pred_meas(state)  # z_hat=H_c*x_hat
    kalman_gain = state.cov @ sens_modl.H.T @ np.linalg.inv(pred_meas.cov)
    innovation = meas.value - pred_meas  # TODO
    cond_mean = state.mean + kalman_gain @ innovation  # p(x|z_c)
    cond_cov = state.cov - \
        kalman_gain @ pred_meas.cov @ kalman_gain.T  # p(x|z_c)

    cond_state = MultiVarGauss2d(cond_mean, cond_cov)

    # # TODO replace this with own code
    # cond_state = conditioning_solu.get_cond_state(state, sens_modl, meas)

    return cond_state
