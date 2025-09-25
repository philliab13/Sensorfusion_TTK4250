from dataclasses import dataclass
from typing import Sequence
from senfuslib import MultiVarGauss, DynamicModel, SensorModel, TimeSequence
import numpy as np
import tqdm
import logging

from states import StateCV, MeasPos
from models import ModelImm
from sensors import SensorPos
from gaussian_mixture import GaussianMixture
from solution import filter as filter_solu


@dataclass
class OutputEKF:
    x_est_upd: MultiVarGauss[StateCV]
    x_est_pred: MultiVarGauss[StateCV]
    z_est_pred: MultiVarGauss[MeasPos]


@dataclass
class EKF:
    dynamic_model: DynamicModel[StateCV]
    sensor_model: SensorModel[MeasPos]

    def step(self,
             x_est_prev: MultiVarGauss[StateCV],
             z: MeasPos,
             dt: float) -> OutputEKF:
        """Perform one EKF update step."""
        x_est_pred = self.dynamic_model.pred_from_est(x_est_prev, dt)
        z_est_pred = self.sensor_model.pred_from_est(x_est_pred)

        H_mat = self.sensor_model.H(x_est_pred.mean)
        P_mat = x_est_pred.cov
        S_mat = z_est_pred.cov

        kalman_gain = P_mat @ H_mat.T @ np.linalg.inv(S_mat)
        innovation = z - z_est_pred.mean

        state_upd_mean = x_est_pred.mean + kalman_gain @ innovation
        state_upd_cov = P_mat - kalman_gain @ H_mat @ P_mat

        x_est_upd = MultiVarGauss(state_upd_mean, state_upd_cov)

        return OutputEKF(x_est_upd, x_est_pred, z_est_pred)


@dataclass
class FilterIMM:
    dynamic_model: ModelImm
    sensor_model: SensorPos

    def calculate_mixings(self, x_est_prev: GaussianMixture[StateCV],
                          dt: float) -> np.ndarray:
        """Calculate the mixing probabilities, following step 1 in (6.4.1).

        The output should be on the following from:
        $mixing_probs[s_{k-1}, s_k] = \mu_{s_{k-1}|s_k}$
        """
        pi_mat = self.dynamic_model.get_pi_mat_d(dt)  # the pi in (6.6)
        prev_weights = x_est_prev.weights  # \mathbf{p}_{k-1}

        joint = pi_mat * prev_weights[:, None]

        col_sums = joint.sum(axis=0, keepdims=True)  # Normalization

        with np.errstate(divide='ignore', invalid='ignore'):
            mixing_probs = np.divide(
                joint, col_sums, out=np.zeros_like(joint), where=col_sums > 0)

        return mixing_probs

    def mixing(self,
               x_est_prev: GaussianMixture[StateCV],
               mixing_probs: GaussianMixture[StateCV]
               ) -> Sequence[MultiVarGauss[StateCV]]:
        """Calculate the moment-based approximations, 
        following step 2 in (6.4.1). 
        Should return a gaussian with mean=(6.34) and cov=(6.35).

        Hint: Create a GaussianMixture for each mode density (6.33), 
        and use .reduce() to calculate (6.34) and (6.35).
        """
        # r = len(x_est_prev)

        # # Accept either ndarray or object with .weights
        # W = getattr(mixing_probs, "weights", mixing_probs)
        # W = np.asarray(W, dtype=float)

        # if W.shape != (r, r):
        #     raise ValueError(
        #         f"mixing_probs must be shape {(r, r)}; got {W.shape}")

        # # Detect / fix orientation:
        # # We want columns j to be μ_{i|j} over previous modes i.
        # col_ok = np.allclose(
        #     W.sum(axis=0, where=np.isfinite(W)), 1.0, atol=1e-6)
        # row_ok = np.allclose(
        #     W.sum(axis=1, where=np.isfinite(W)), 1.0, atol=1e-6)
        # if not col_ok and row_ok:
        #     W = W.T  # it was μ_{j|i}; make it μ_{i|j}

        # # Normalize columns safely
        # W = np.clip(W, 0.0, None)
        # col_sums = W.sum(axis=0, keepdims=True)
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     W = np.divide(W, col_sums, out=np.zeros_like(W),
        #                   where=col_sums > 0)

        # # Fallback if a column is all-zero (unreachable next-mode j):
        # prev_w = np.asarray(x_est_prev.weights, dtype=float)
        # prev_w = prev_w / prev_w.sum() if prev_w.sum() > 0 else np.full(r, 1.0/r)

        # moment_based_preds: list[MultiVarGauss[StateCV]] = []
        # for j in range(r):
        #     w_j = W[:, j]
        #     if not np.isfinite(w_j).all() or w_j.sum() == 0:
        #         w_j = prev_w

        #     # Build the exact mixture: sum_i μ_{i|j} N(·; x̂_{k-1}^{(i)}, P_{k-1}^{(i)})
        #     mixture = GaussianMixture(w_j, x_est_prev.gaussians)  # (6.33)

        #     # Moment-match → (6.34)-(6.35)
        #     moment_based_pred = mixture.reduce()  # -> MultiVarGauss
        #     moment_based_preds.append(moment_based_pred)
        moment_based_preds = []
        for i in range(len(x_est_prev)):
            # Mixture is step (6.33)
            # Here the sum is implicite in creating the mixture
            mixture = GaussianMixture(mixing_probs[:, i], x_est_prev.gaussians)
            # Use mixture reduce, since that was in the hint
            moment_based_pred = mixture.reduce()
            # Then add all the different models for each and every mode s_k into a vector and return
            moment_based_preds.append(moment_based_pred)

        return moment_based_preds

    def mode_match_filter(self,
                          moment_based_preds: GaussianMixture[StateCV],
                          z: MeasPos,
                          dt: float
                          ) -> Sequence[OutputEKF]:
        """Calculate the mode-match filter outputs (6.36),
        following step 3 in (6.4.1). 

        Hint: Use the EKF class from the top of this file.
        The last part (6.37) is not part of this
        method and is done later."""
        ekf_outs = []

        for i, x_prev in enumerate(moment_based_preds):
            ekf = EKF(self.dynamic_model.models[i], self.sensor_model)
            out_ekf = ekf.step(x_prev, z, dt)  # TODO (OutputEKF)
            ekf_outs.append(out_ekf)

        return ekf_outs

    def update_probabilities(self,
                             ekf_outs: Sequence[OutputEKF],
                             z: MeasPos,
                             dt: float,
                             weights: np.ndarray
                             ) -> np.ndarray:
        """Update the mixing probabilities,
        using (6.37) from step 3 and (6.38) from step 4 in (6.4.1).

        Hint: Use (6.6)
        """
        pi = self.dynamic_model.get_pi_mat_d(
            dt)  # π[i,j] = P(s_k=j | s_{k-1}=i)

        # (6.6) predict mode probabilities
        w_pred = pi.T @ weights

        lik = np.array([out.z_est_pred.pdf(z)
                       for out in ekf_outs], dtype=float)
        lik = np.clip(lik, 0.0, None)  # guard tiny negatives from numerics

        # (6.38) update and normalize
        w_post = w_pred * lik
        s = w_post.sum()
        w_post /= s

        return w_post

    def step(self,
             x_est_prev: GaussianMixture[StateCV],
             z: MeasPos,
             dt) -> GaussianMixture[StateCV]:
        """Perform one step of the IMM filter."""
        mixing_probs = self.calculate_mixings(x_est_prev, dt)  # TODO step 1
        moment_based_preds = self.mixing(
            x_est_prev, mixing_probs)  # TODO step 2
        ekf_outs = self.mode_match_filter(
            moment_based_preds, z, dt)  # TODO step 3
        weights_upd = self.update_probabilities(
            ekf_outs, z, dt, x_est_prev.weights)  # TODO step 4
        x_est_upd = GaussianMixture(
            weights_upd, [out.x_est_upd for out in ekf_outs])
        if ekf_outs is not None:  # You can remove this
            x_est_pred = GaussianMixture(x_est_prev.weights,
                                         [out.x_est_pred for out in ekf_outs])

            z_est_pred = GaussianMixture(x_est_prev.weights,
                                         [out.z_est_pred for out in ekf_outs])

        return x_est_upd, x_est_pred, z_est_pred

    def run(self, x0_est: GaussianMixture[StateCV], zs: TimeSequence[MeasPos]
            ) -> TimeSequence[GaussianMixture[StateCV]]:
        """Run the IMM filter."""
        logging.info("Running IMM filter")
        x_est_upds = TimeSequence()
        x_est_preds = TimeSequence()
        z_est_preds = TimeSequence()
        x_est_upds.insert(0, x0_est)
        t_prev = 0
        for t, z in tqdm.tqdm(zs.items(), total=len(zs)):
            t_prev, x_est_prev = x_est_upds[-1]
            dt = np.round(t-t_prev, 8)

            x_est_upd, x_est_pred, z_est_pred = self.step(x_est_prev, z, dt)
            x_est_upds.insert(t, x_est_upd)
            x_est_preds.insert(t, x_est_pred)
            z_est_preds.insert(t, z_est_pred)
        return x_est_upds, x_est_preds, z_est_preds
