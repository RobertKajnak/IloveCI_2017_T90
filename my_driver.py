from pytocl.driver import Driver
from sklearn.linear_model import Ridge
from pytocl.car import State, Command, MPS_PER_KMH
import logging
import pickle
import math
from sklearn.externals import joblib
import numpy as np

with open('ESN0.pkl', 'rb') as ESN_pickle:
    my_esn = joblib.load(ESN_pickle)

with open('regr.pkl', 'rb') as regr_pickle:
    regr = joblib.load(regr_pickle)
   
class MyDriver(Driver):
    # Override the `drive` method to create your own driver

    def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
        command = Command()
        input = [carstate.speed_x, carstate.distance_from_center, carstate.angle] \
        + list(carstate.distances_from_edge)
        echo_input = np.atleast_2d(input)
        echo_input = my_esn.fit_transform(echo_input)
        echo_output = regr.predict(echo_input)
        if echo_output[0,0] > 1:
            accelerate = 1
        elif echo_output[0,0] < 1:
            accelerate = 0
        else:
            accelerate = echo_output[0,0]
        if echo_output[0,1] > 1:
            brake = 1
        elif echo_output[0,1] < 1:
            brake = 0
        else:
            brake = echo_output[0,1]
        print(accelerate, brake, echo_output[0,2])
        command.accelerator = accelerate
        command.brake = brake
        command.steering = echo_output[0, 2]
                  
        #command.accelerator or acceleration?
                 
        #command.accelerator = 1
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500 and carstate.gear > 0:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
        
        print(command.gear)    
            
        return command
 
