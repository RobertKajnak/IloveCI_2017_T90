"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
from numpy import loadtxt, atleast_2d
import numpy as np
from sklearn.externals import joblib
import subprocess
import csv
#import visualize

# 2-input XOR inputs and expected outputs.
#xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
#xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

X1 = loadtxt('torcs-client/aalborg.csv',  delimiter=",", skiprows=1)
T = loadtxt('torcs-client/alpine-1.csv', delimiter=",", skiprows=1)
X2 = loadtxt('torcs-client/f-speedway.csv', delimiter=",", skiprows=1)
data = np.concatenate((X1, X2, T))



xor_inputs = data[:, 3:] 
xor_outputs = data[:, :3]


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        # for now: assume only 10 drivers
        i = genome_id - 1
        filename = "torcs-client/genomes/genome" + str(i) + "/genome.pkl"
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        joblib.dump(net, filename) 
        genome.fitness = 1300

    subprocess.check_output(["python3", "torcs_tournament.py", "quickrace.yml"])
    with open("ratings.csv") as ratingfile:
        reader = csv.reader(ratingfile, delimiter=',')
        ratings = list(reader)
        for genome_id, genome in genomes:
            for rating in ratings:
                if rating[0] == "student" + str(genome_id):
                    fitness = rating[1]
            print(fitness)
            genome.fitness = float(fitness)

#        genome.fitness = 4.0
#        net = neat.nn.FeedForwardNetwork.create(genome, config)
#        for xi, xo in zip(xor_inputs, xor_outputs):
#            output = net.activate(xi)
#            genome.fitness -= (output[0] - xo[0]) ** 2


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 3)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
#    print('\nOutput:')
#    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
#    for xi, xo in zip(xor_inputs, xor_outputs):
#        output = winner_net.activate(xi)
#        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
#    visualize.draw_net(config, winner, True, node_names=node_names)
#    visualize.plot_stats(stats, ylog=False, view=True)
#    visualize.plot_species(stats, view=True)

#    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
#    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
