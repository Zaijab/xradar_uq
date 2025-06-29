"""
This is an example script which executes the entire filtering loop and calculates the ST-RMSE.
"""

"""
"""
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import RangeSensor
from xradar_uq.stochastic_filters import EnKF

"""
Now we shall instantiate the objects we imported.
They will constitute our state space system.
"""


"""
First, the dynamical system object.
"""
dynamical_system = CR3BP()

"""

"""
measurement_system = RangeSensor()

stochastic_filter = EnKF()
