"""Behavior-state HMM (hmmlearn with Markov-chain fallback)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class BehaviorHMM:
    """Hidden Markov Model for behavior state transitions.

    Falls back to a simple Markov chain (transition matrix) if hmmlearn is
    unavailable at import time. This matches the old backend's strategy.
    """

    STATE_NAMES = ["deep_focus", "shallow_work", "browsing", "procrastination", "idle"]

    def __init__(self, n_states: int = 5) -> None:
        self.n_states = n_states
        self.model: Any = None
        self.state_names = self.STATE_NAMES[:n_states]
        self.transition_matrix: npt.NDArray[Any] | None = None
        self._is_fitted: bool = False

    def fit(self, sequences: list[npt.NDArray[Any]]) -> BehaviorHMM:
        """Fit HMM to sequences of state observations.

        Args:
            sequences: list of 1D arrays, each containing a sequence of
                state IDs (0-indexed).

        Returns:
            self
        """
        self.transition_matrix = self._compute_transition_matrix(sequences)

        try:
            import hmmlearn.hmm as hmm

            X, lengths = self._prepare_hmm_data(sequences)
            if X is not None and len(X) >= 10:
                self.model = hmm.CategoricalHMM(
                    n_components=self.n_states,
                    random_state=42,
                    n_iter=100,
                    tol=1e-4,
                )
                self.model.fit(X, lengths)
        except ImportError:
            self.model = None

        self._is_fitted = True
        return self

    def _compute_transition_matrix(self, sequences: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
        """Compute Markov transition matrix from state sequences."""
        matrix = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        counts = np.zeros((self.n_states, self.n_states), dtype=np.int32)

        for seq in sequences:
            for i in range(len(seq) - 1):
                s_from = int(seq[i])
                s_to = int(seq[i + 1])
                if 0 <= s_from < self.n_states and 0 <= s_to < self.n_states:
                    counts[s_from, s_to] += 1

        for i in range(self.n_states):
            row_sum = counts[i].sum()
            if row_sum > 0:
                matrix[i] = counts[i] / row_sum
            else:
                matrix[i] = np.ones(self.n_states) / self.n_states

        return matrix

    @staticmethod
    def _prepare_hmm_data(
        sequences: list[npt.NDArray[Any]],
    ) -> tuple[npt.NDArray[Any] | None, npt.NDArray[Any] | None]:
        """Prepare data for hmmlearn format."""
        all_observations: list[int] = []
        lengths: list[int] = []
        for seq in sequences:
            n_states = 5  # default; caller should ensure state IDs are valid
            filtered = [int(s) for s in seq if 0 <= int(s) < n_states]
            if filtered:
                all_observations.extend(filtered)
                lengths.append(len(filtered))
        if not all_observations:
            return None, None
        lengths_array = np.array(lengths, dtype=np.int32)
        X = np.array(all_observations).reshape(-1, 1)
        return X, lengths_array

    def predict_next_state(self, current_state: int) -> dict[str, Any]:
        """Predict most likely next state and probability distribution.

        Args:
            current_state: current state ID (0-indexed).

        Returns:
            dict with keys: ``next_state`` (int), ``probabilities`` (list[float]),
            ``next_state_name`` (str).
        """
        if not self._is_fitted or self.transition_matrix is None:
            uniform = 1.0 / self.n_states
            return {
                "next_state": 0,
                "probabilities": [uniform] * self.n_states,
                "next_state_name": self.state_names[0],
            }

        probs = self._get_transition_probs(current_state)
        next_state = int(np.argmax(probs))

        return {
            "next_state": next_state,
            "probabilities": [round(float(p), 4) for p in probs],
            "next_state_name": self.state_names[next_state],
        }

    def _get_transition_probs(self, state: int) -> npt.NDArray[Any]:
        """Get transition probabilities from hmmlearn or matrix."""
        if self.model is not None:
            try:
                transmat = self.model.transmat_
                if 0 <= state < transmat.shape[0]:
                    return np.asarray(transmat[state])
            except (AttributeError, IndexError):
                pass

        if self.transition_matrix is not None and 0 <= state < self.n_states:
            return np.asarray(self.transition_matrix[state])

        return np.ones(self.n_states) / self.n_states

    def get_transition_matrix(self) -> npt.NDArray[Any]:
        """Return the transition matrix as a 2D numpy array."""
        if self.transition_matrix is None:
            return np.ones((self.n_states, self.n_states)) / self.n_states
        return self.transition_matrix

    def get_steady_state(self) -> npt.NDArray[Any]:
        """Compute steady-state distribution via eigenvector of transition matrix.

        Returns probability distribution over states.
        """
        mat = self.get_transition_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(mat.T)
        idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
        steady = np.asarray(np.real(eigenvectors[:, idx]))
        steady = steady / steady.sum()
        return np.asarray(np.clip(steady, 0.0, 1.0))
