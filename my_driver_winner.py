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
   
with open('winnernet.pkl', 'rb') as genome_pickle:
        genome = joblib.load(genome_pickle)
  
class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    my_esn = None;
    regr=None;
    transTimer = Timer('Transform')
    regrTimer = Timer('regr')
    
    def __init__(self):

        t = Timer();
#        print("It gets here")
#        with open('mlp.pkl', 'rb') as ESN_pickle:
#            self.my_esn = joblib.load(ESN_pickle)

#        with open('regr.pkl', 'rb') as regr_pickle:
#            self.regr = joblib.load(regr_pickle)
        t.clock()
        
    def drive(self,carstate: State) -> Command:
        command = Command()
        
        input = [carstate.speed_x, carstate.distance_from_center, carstate.angle] \
        + list(carstate.distances_from_edge)
        
        d_input = np.atleast_2d(input)

        output = genome.activate(input)

        accelerate = round(output[0])
        brake = round(output[1])
        steer = output[2]
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

