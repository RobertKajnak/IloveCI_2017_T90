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
    
    def __init__(self):
        with open('mlp.pkl', 'rb') as ESN_pickle:
            self.my_esn = joblib.load(ESN_pickle)

        with open('regr.pkl', 'rb') as regr_pickle:
            self.regr = joblib.load(regr_pickle)
        self.last_angle = 2000
        
        #crash detection
        self.is_frontal=0
        self.frontal_start = 0
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
            #print('full braking')

        command.accelerator = accelerate
        command.brake = brake
        command.steering = steer
          
     
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 2500 and carstate.gear > 0:
            command.gear = carstate.gear - 1
                

        if not command.gear:
            command.gear = carstate.gear or 1

        #frontal crash detected
        #! NEEDS TO GO AFTER GEARCHANGE!
        d_front = carstate.distances_from_edge[9]      
        d_center = carstate.distance_from_center
        #print(carstate.current_lap_time - self.frontal_start)
        if carstate.speed_x<2 and carstate.speed_x>0 and d_front<2 and \
                carstate.current_lap_time - self.frontal_start >3:
            self.is_frontal = -1 if d_center<0 else 1
            self.frontal_start = carstate.current_lap_time
            self.frontal_dist = abs(d_center)*1.3

        if self.is_frontal!=0:
            print("Reversing" + str(self.frontal_dist))
            command.accelerator = 0.7
            command.steering = self.is_frontal 
            if self.frontal_dist>3:
                command.steering *= 0.5
            
            command.gear = -1   
            if carstate.current_lap_time - self.frontal_start > self.frontal_dist:
                self.is_frontal = False
                command.gear = 1
                
       
        #print(accelerate, brake, steer, carstate.angle)
             
        return command
