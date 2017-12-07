import logging

import math

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController

_logger = logging.getLogger(__name__)


class Driver:
    """
    Driving logic.

    Implement the driving intelligence in this class by processing the current
    car state as inputs creating car control commands as a response. The
    ``drive`` function is called periodically every 20ms and must return a
    command within 10ms wall time.
    """

    def __init__(self, logdata=True):
            
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )

        self.steerers = [CompositeController(ProportionalController(0.4*i),
            IntegrationController(0.2*i, integral_limit=1.5),
            DerivativeController(2/i)) for i in range(1,5)]
        
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None

    @property
    def range_finder_angles(self):
        """Iterable of 19 fixed range finder directions [deg].

        The values are used once at startup of the client to set the directions
        of range finders. During regular execution, a 19-valued vector of track
        distances in these directions is returned in ``state.State.tracks``.
        """
        return -90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, \
            30, 45, 60, 75, 90

    def on_shutdown(self):
        """
        Server requested driver shutdown.

        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None

    def speed_by_dist(self,d):
        if d>=150:
            y=400
        elif d>30:
            y=100+2.5*(d-30)
        elif d>20:
            y=105
        else:
            y=55    
        
        return y
        
    def speed_category(self,v):
        cat = 0        
        if v<40:
            cat = 0
        elif v<100:
            cat=1
        elif v<200:
            cat = 2
        else:
            cat = 3
        return cat
    
    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        command = Command()
        self.steer(carstate, 0.0, command)

        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        #v_x = 300
        v_x = self.speed_by_dist(carstate.distances_from_edge[9])

        #print(carstate.distances_from_edge[9],v_x)

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command

    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        else:
            command.brake = min(-acceleration, 1)

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        command.steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )
        
        '''for steerer in self.steerers:
            steerer.control(steering_error,carstate.current_lap_time)
        command.steering = self.steerers[self.speed_category(carstate.speed_x)].\
                control(steering_error,carstate.current_lap_time)'''
        
    def complex_control(self, carstate, command):
        
        return 