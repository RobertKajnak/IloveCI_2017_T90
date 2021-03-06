from pytocl.driver import Driver
from pytocl.car import State, Command
from sklearn.externals import joblib
import numpy as np
import time
import csv
# based on my_driver_heuristics
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

        self.last_angle = 2000

        
        # pick idenity and ensure that all files exist
        try: 
            with open('position_team90.csv', 'r+') as csvfile: 
                reader = csv.reader(csvfile)
                row = next(reader)
                if 'a' in row:
                    self.identity = 'b'
                    with open('position_team90.csv', 'w') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        writer.writerow('b')
                else:
                    self.identity = 'a'
                    with open('position_team90.csv', 'w') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        writer.writerow('a')   
        except OSError:
            self.identity = 'a'
            with open('position_team90.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow('a') 
        open('positiona_team90.csv', 'w')
        open('positionb_team90.csv', 'w')
        #crash detection
        self.is_frontal=0
        self.frontal_start = 0
        self.frontal_dist =0
        self.last_gear = 0 
        self.last_shift_time = -5

    #Prevents shifting oscillation        
    #up_down = gear shift amount: positive for up, negative for down
    def is_safe_shift(self,carstate,attempted_gear,time_limit=1.5):
        return not ((self.last_gear == attempted_gear) and \
                (carstate.current_lap_time-time_limit-self.last_shift_time>0))

        
    def drive(self,carstate: State) -> Command:
        print(self.identity)
        command = Command()
        
        input = [carstate.speed_x, carstate.distance_from_center, carstate.angle] \
        + list(carstate.distances_from_edge)
        
        d_input = np.atleast_2d(input)
        steer = mlp.predict(d_input)[0,2]/10
    
        echo_input = my_esn.transform(d_input)
        output = regr.predict(echo_input)

        # Cap acceleration and braking between 0 and 1
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
        
        # If car goes off track, but still roughly following the track, steer to get it back on track
        if carstate.distance_from_center > 1 and abs(carstate.angle) < 20:
            steer = -0.1 
        elif carstate.distance_from_center < -1 and abs(carstate.angle) < 20:
            steer = 0.1
        
        # Full steer in car not following the track
        if carstate.angle > 50:
            steer = 1
        elif carstate.angle < -50:
            steer = -1
        
        # Full steer in car stays in same angle too long (e.g. following a wall)
        elif carstate.angle > 10 and abs(self.last_angle - carstate.angle) < 0.5:
            steer = 1
        elif carstate.angle < -10 and abs(self.last_angle - carstate.angle) < 0.5:
            steer = -1
        self.last_angle = carstate.angle
        
        # Brake when taking a corner to avoid going too fast    
        if abs(steer) > 0.05 and carstate.speed_x > 20:
            brake = 1
            #print('full braking')

        d_front = carstate.distances_from_edge[9]      
        d_center = carstate.distance_from_center
        if carstate.speed_x<2 and carstate.speed_x>0 and d_front<2 and \
                carstate.current_lap_time - self.frontal_start >self.frontal_dist*3:
            self.is_frontal = -1 if d_center<0 else 1
            self.frontal_start = carstate.current_lap_time
            self.frontal_dist = abs(d_center)*1.3

        # Get files to read and write
        if self.identity == 'a':
            this_file = 'positiona_team90.csv'
            other_file = 'positionb_team90.csv'
        else:
            this_file = 'positionb_team90.csv'
            other_file = 'positiona_team90.csv'

        # Different behavior if behind
        behind = False
        with open(other_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                rowlist = list(row)
                if len(rowlist) > 1 and int(rowlist[1]) < carstate.race_position:
                    behind = True
                else:
                    behind = False
        if behind and self.is_frontal == 0:
            min_index = np.argmin(carstate.opponents)
            car_distance = list(carstate.opponents)[min_index]
            # Do not steer if cars are too far away or closest car is directly behind
            if car_distance < 50 and min_index not in range(0,6) and min_index not in range(30, 36): 
                degrees = -180 + 10*min_index
                print(degrees)
                accelerate = 1
                brake = 0
                # full steer is 21 degrees
                if -21 < degrees < 21:
                    print('steering to car')
                    steer = -degrees/21
                elif degrees < -21:
                    print('small degrees')
                    steer = 1
                else:
                    print('large degrees')
                    steer = -1    

        command.accelerator = accelerate
        command.brake = brake
        command.steering = steer
        #print(accelerate, brake, steer)
        #command.accelerator or acceleration?
        
        #command.accelerator = 1
     
        if carstate.rpm > 8000 and self.is_safe_shift(carstate,carstate.gear+1):
            print('gear up')
            command.gear = carstate.gear + 1
        if carstate.rpm < 2500 and carstate.gear > 0 and self.is_safe_shift(carstate,carstate.gear-1):
            print('gear down')
            command.gear = carstate.gear - 1
               

        with open(this_file, 'w') as csvfile: 
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.identity, carstate.race_position, carstate.distance_raced]) 

        #frontal crash detected
        #! NEEDS TO GO AFTER GEARCHANGE!
#        d_front = carstate.distances_from_edge[9]      
#        d_center = carstate.distance_from_center
        
#        if carstate.speed_x<2 and carstate.speed_x>0 and d_front<2 and \
#                carstate.current_lap_time - self.frontal_start >self.frontal_dist*3:
#            self.is_frontal = -1 if d_center<0 else 1
#            self.frontal_start = carstate.current_lap_time
#            self.frontal_dist = abs(d_center)*1.3

        if self.is_frontal!=0:
            print("Reversing" + str(self.frontal_dist))
            command.accelerator = 0.7
            command.steering = self.is_frontal 
            if self.frontal_dist>3:
                command.steering *= 0.5
            
            command.gear = -1   
            if carstate.current_lap_time - self.frontal_start > self.frontal_dist:
                print('changed gear to one')
                self.is_frontal = 0
                command.gear = 1

        if not command.gear:
            print('keep gear same')
            command.gear = carstate.gear or 1
        if self.is_frontal == 0 and command.gear == -1:
            command.gear = 1
        print(command.accelerator, command.brake, command.steering, command.gear)

        return command



