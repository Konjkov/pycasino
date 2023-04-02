from abc import ABC, abstractmethod
import numpy as np


class Slater(ABC):

    @abstractmethod
    def value(self, n_vectors: np.ndarray) -> float:
        """Value φ(r)"""

    @abstractmethod
    def gradient(self, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient ∇φ(r)/φ(r) w.r.t e-coordinates."""

    @abstractmethod
    def laplacian(self, n_vectors: np.ndarray) -> float:
        """Laplacian Δφ(r)/φ(r) w.r.t e-coordinates."""

    @abstractmethod
    def hessian(self, n_vectors: np.ndarray) -> np.ndarray:
        """Hessian H(φ(r))/φ(r) w.r.t e-coordinates."""


class Jastrow(ABC):

    @abstractmethod
    def value(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> float:
        """Value"""

    @abstractmethod
    def gradient(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient w.r.t e-coordinates."""

    @abstractmethod
    def laplacian(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> float:
        """Laplacian w.r.t e-coordinates."""

    @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Get parameters"""

    @abstractmethod
    def set_parameters(self, parameters: np.ndarray):
        """Set parameters"""


class Backflow(ABC):

    @abstractmethod
    def value(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        """Value"""

    @abstractmethod
    def gradient(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient w.r.t e-coordinates."""

    @abstractmethod
    def laplacian(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        """Laplacian w.r.t e-coordinates."""

    @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Get parameters"""

    @abstractmethod
    def set_parameters(self, parameters: np.ndarray):
        """Set parameters"""
