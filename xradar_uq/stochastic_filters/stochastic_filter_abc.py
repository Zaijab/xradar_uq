import abc

import equinox as eqx
from jaxtyping import Array, Float, Key
from xradar_uq.measurement_systems import AbstractMeasurementSystem


class AbstractFilter(eqx.Module, strict=True):
    """
    This is an abstract base class representing the functionality of a filter.
    Stochastic filters assume three parts:

    - Initialization: What is the initial guess of my state?
    - Prediction: Where do I expect the state to go next?
    - Update: 
    """

    measurement_system: AbstractMeasurementSystem
    dynamical_system: AbstractDynamicalSystem
    
    @abc.abstractmethod
    def initialize(
        self,
        key: Key[Array, "..."],
    ) -> Float[Array, "batch_size state_dim"]:
        raise NotImplementedError
    
    
    @abc.abstractmethod
    def predict(
        self,
        key: Key[Array, "..."],
        posterior_ensemble: Float[Array, "batch_size state_dim"],
        measurement: Float[Array, "measurement_dim"],
    ) -> Float[Array, "batch_size state_dim"]:
        raise NotImplementedError
    

    @abc.abstractmethod
    def update(
        self,
        key: Key[Array, "..."],
        prior_ensemble: Float[Array, "batch_size state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_size state_dim"]:
        """
        Given some noisy measurement and my current understanding of the state, how should I update my degrees of beliefs?
        """
        raise NotImplementedError
