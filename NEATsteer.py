"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
from numpy import loadtxt, atleast_2d
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import warnings
import matplotlib.pyplot as plt


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

# 2-input XOR inputs and expected outputs.
#xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
#xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

X1 = loadtxt('aalborg.csv',  delimiter=",", skiprows=1)
T = loadtxt('alpine-1.csv', delimiter=",", skiprows=1)
X2 = loadtxt('f-speedway.csv', delimiter=",", skiprows=1)
data = np.concatenate((X1, X2, T))

xor_inputs = data[:50, 3:] 
xor_outputs = data[:50, 2:3]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 50
        net = neat.nn.RecurrentNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
#            if output[0] > 0 and output[0] > 0.15 and xo[0] < 0 and xo[0]>0.15:
#                fitness = 1
#            if output[0] > 0 and output[0] < 0.15 and xo[0] < 0 and xo[0]<0.15:
#                fitness = 1
#        
            if xo[0] < 0 and output[0] >0:
                penalty = 1
            elif xo[0] > 0 and output[0]<0:
                penalty = 1
            else:
                penalty = 0
            #if output
              
            genome.fitness -= (((output[0]) - (xo[0])) **2) + penalty
            print (output)
            print (xo)
            print (penalty)
            print (((output[0]) - (xo[0]) **2))
            print (genome.fitness)
           # print (penalty)
            #output = net.activate(xi)
            #genome.fitness += 1 - mean_squared_error(output, xo)
           #genome.fitness -= (output[0] - xo[0]) ** 2
            
# err = mean_squared_error(y_true, y_pred)


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
    winner = p.run(eval_genomes, 5)
    joblib.dump(winner, 'winner.pkl') 

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
       # print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
        
    joblib.dump(winner_net, 'winnernet.pkl') 

#    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
#    visualize.draw_net(config, winner, True, node_names=node_names)
    plot_stats(stats, ylog=False, view=True)
#    visualize.plot_species(stats, view=True)

#    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
 #   p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-steer')
    run(config_path)



