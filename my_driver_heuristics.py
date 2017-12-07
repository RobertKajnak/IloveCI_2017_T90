from pytocl.driver import Driver
from sklearn.linear_model import Ridge
from pytocl.car import State, Command, MPS_PER_KMH
import logging
import pickle
import math
from sklearn.externals import joblib
import numpy as np
import time
from collections import deque

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
    
    def __init__(self):
        t = Timer();
        with open('mlp.pkl', 'rb') as ESN_pickle:
            self.my_esn = joblib.load(ESN_pickle)

        with open('regr.pkl', 'rb') as regr_pickle:
            self.regr = joblib.load(regr_pickle)
        self.last_angle = 2000
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
        
        if carstate.distance_from_center > 1 and abs(carstate.angle) < 20: # off track left
            steer = -0.1 
        elif carstate.distance_from_center < -1 and abs(carstate.angle) < 20:
            steer = 0.1
#        elif 0.8 < carstate.distance_from_center < 1 and abs(carstate.angle) < 20:
#            steer = -0.04
#        elif -0.8 > carstate.distance_from_center > -1and abs(carstate.angle) < 20:
#            steer = 0.04
        if carstate.angle > 50:
            steer = 1
        elif carstate.angle < -50:
            steer = -1
        elif carstate.angle > 10 and abs(self.last_angle - carstate.angle) < 0.5:
            steer = 1
        elif carstate.angle < -10 and abs(self.last_angle - carstate.angle) < 0.5:
            steer = -1
        self.last_angle = carstate.angle
            
        if abs(steer) > 0.05 and carstate.speed_x > 20:
            brake = 1
            print('full braking')

        command.accelerator = accelerate
        command.brake = brake
        command.steering = steer
        #print(accelerate, brake, steer)
        #command.accelerator or acceleration?
        
        #command.accelerator = 1
     
        if carstate.rpm > 4000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 2500 and carstate.gear > 0:
            command.gear = carstate.gear - 1
                

        if not command.gear:
            command.gear = carstate.gear or 1
        
       
        print(accelerate, brake, steer, carstate.angle)
             
        return command
