from pytocl.driver import Driver
from sklearn.linear_model import Ridge
from pytocl.car import State, Command, MPS_PER_KMH
import logging
import pickle
import math
from sklearn.externals import joblib
import numpy as np

with open('ESN0.pkl', 'rb') as pickle_file:
    my_esn = joblib.load(pickle_file)
   

class MyDriver(Driver):
    # Override the `drive` method to create your own driver

    def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
         command = Command()
         regr = Ridge(alpha = 0.01)
         input = [carstate.speed_x, carstate.distance_from_center, carstate.angle] \
         + list(carstate.distances_from_edge)
         echo_input = np.atleast_2d(input)
         echo_input = my_esn.fit_transform(echo_input)
         echo_output = regr.predict(echo_input)
         command.accelerator = echo_output[0]
         command.brake = echo_output[1]
         command.steering = echo_output[2]
                  
         #command.accelerator or acceleration?
                 
         #command.accelerator = 1
         if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

         if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

         if not command.gear:
            command.gear = carstate.gear or 1
            
            
         return command
 
