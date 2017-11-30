#! /usr/bin/env python3
import sys
sys.path.insert(0, '/home/student/Documents/torcs-server/torcs-client')
from pytocl.main import main
#from my_driver import MyDriver
#from my_driver_combi import MyDriver
from my_driver_genome import MyDriver

import argparse


if __name__ == '__main__':
    main(MyDriver())


