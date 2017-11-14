from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
import logging

import math

class MyDriver(Driver):
    # Override the `drive` method to create your own driver

    def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
         command = Command()
         command.accelerator = 1
         if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

         if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

         if not command.gear:
            command.gear = carstate.gear or 1
            
            
         return command
 