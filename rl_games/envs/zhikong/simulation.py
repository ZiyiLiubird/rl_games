import jsbsim
import os
import time
from mpl_toolkits.mplot3d import Axes3D  # req'd for 3d plotting
from typing import Dict, Union
import zhikong.properties as prp
from zhikong.aircraft import Aircraft, cessna172P
from zhikong.properties import Property, BoundedProperty
import numpy as np
import socket
import json

class Simulation(object):
    """
    A class which wraps an instance of JSBSim and manages communication with it.
    """


    def __init__(self,aircraft: Aircraft = cessna172P):
        """
        Constructor. Creates an instance of JSBSim and sets initial conditions.

        :param sim_frequency_hz: the JSBSim integration frequency in Hz.
        :param aircraft_model_name: name of aircraft to be loaded.
            JSBSim looks for file \model_name\model_name.xml from root dir.
        :param init_conditions: dict mapping properties to their initial values.
            Defaults to None, causing a default set of initial props to be used.
        :param allow_flightgear_output: bool, loads a config file instructing
            JSBSim to connect to an output socket if True.
        """
        substitution_name = [it[1].name for it in vars(prp).items() if isinstance(it[1], (Property, BoundedProperty))]
        initial_dict_value = np.zeros(len(substitution_name))
        self.substitution_dict = dict(zip(substitution_name, initial_dict_value))
        self.aircraft = aircraft



    def __getitem__(self, prop: Union[prp.BoundedProperty, prp.Property]) -> float:
        """
        Retrieves specified simulation property.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        :param prop: BoundedProperty, the property to be retrieved
        :return: float
        """
        return self.substitution_dict[prop.name]

    def __setitem__(self, prop: Union[prp.BoundedProperty, prp.Property], value) -> None:
        """
        Sets simulation property to specified value.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        Warning: JSBSim will create new properties if the specified one exists.
        If the property you are setting is read-only in JSBSim the operation
        will silently fail.

        :param prop: BoundedProperty, the property to be retrieved
        :param value: object?, the value to be set
        """
        if prop.name in self.substitution_dict.keys():
            self.substitution_dict[prop.name] = value
        else:
            self.substitution_dict[prop.name]=value

    def get_aircraft(self) -> Aircraft:
        """
        Gets the Aircraft this sim was initialised with.
        """
        return self.aircraft


    def get_sim_time(self) -> float:
        """ Gets the simulation time from JSBSim, a float. """
        return self[prp.sim_time_s]



    def run(self, control_commands:Dict[Union['prp.BoundedProperty',str],Union[float,str]]) -> bool:

      pass


