from pytocl.driver import Driver
from sklearn.linear_model import Ridge
from pytocl.car import State, Command, MPS_PER_KMH
import logging
import pickle
import math
from sklearn.externals import joblib
import numpy as np
import time

class Timer:
    ctime=0;
    name = ''
    def __init__(self,name=''):
        self.name = name
        self.ctime = time.time()
        
    def reset(self):   
        self.ctime = time.time()        
        
    def clock(self):
        newtime = time.time()
        print('%s took %f ms' %(self.name,(newtime-self.ctime)*1000))
        # self.ctime = newtime

with open('ESN0.pkl', 'rb') as ESN_pickle:
    my_esn = joblib.load(ESN_pickle)

with open('regr.pkl', 'rb') as regr_pickle:
    regr = joblib.load(regr_pickle)
   
with open('mlp.pkl', 'rb') as mlp_pickle:
    mlp = joblib.load(mlp_pickle)
  
class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    my_esn = None;
    regr=None;
    transTimer = Timer('Transform')
    regrTimer = Timer('regr')
    
    def __intit__(self):
        t = Timer();
        with open('mlp.pkl', 'rb') as ESN_pickle:
            self.my_esn = joblib.load(ESN_pickle)

        with open('regr.pkl', 'rb') as regr_pickle:
            self.regr = joblib.load(regr_pickle)
        t.clock()
        
    def drive(self,carstate: State) -> Command:
        command = Command()
        
        input = [carstate.speed_x, carstate.distance_from_center, carstate.angle] \
        + list(carstate.distances_from_edge)
        
        d_input = np.atleast_2d(input)
        steer = mlp.predict(d_input)[0,2]/10

        echo_input = my_esn.transform(d_input)
        output = regr.predict(echo_input)

        if output[0,0] > 1:
            accelerate = 1
        elif output[0,0] < 0:
            accelerate = 0
        else:
            accelerate = output[0,0]
        if output[0,1] > 1:
            brake = 1
        elif output[0,1] < 0:
            brake = 0
        else:
            brake = output[0,1]
        print(accelerate, brake, steer)
        command.accelerator = accelerate
        command.brake = brake
        command.steering = steer
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
        
        
#    def drive(self, carstate: State) -> Command:
#        t = Timer('Input');
#        print('command recieved')
#    #     # Interesting stuff
#        command = Command()
#        t.clock()
#        input = [carstate.speed_x, carstate.distance_from_center, carstate.angle] \
#        + list(carstate.distances_from_edge)
#        echo_input = np.atleast_2d(input)
#        
#        self.transTimer.reset()
#        echo_input = my_esn.transform(echo_input)
#        self.transTimer.clock()
#
#        self.regrTimer.reset()
#        echo_output = regr.predict(echo_input)
#        self.regrTimer.clock()
#        
#        if echo_output[0,0] > 1:
#            accelerate = 1
#        elif echo_output[0,0] < 1:
#            accelerate = 0
#        else:
#            accelerate = echo_output[0,0]
#        if echo_output[0,1] > 1:
#            brake = 1
#        elif echo_output[0,1] < 1:
#            brake = 0
#        else:
#            brake = echo_output[0,1]
#        print(accelerate, brake, echo_output[0,2])
#        command.accelerator = accelerate
#        command.brake = brake
#        command.steering = echo_output[0, 2]
#                  
#        #command.accelerator or acceleration?
#                 
#        #command.accelerator = 1
#        if carstate.rpm > 8000:
#            command.gear = carstate.gear + 1
#
#        if carstate.rpm < 2500 and carstate.gear > 0:
#            command.gear = carstate.gear - 1
#
#        if not command.gear:
#            command.gear = carstate.gear or 1
#        
#        print(command.gear)   
#        return command         
## 
