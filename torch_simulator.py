import numpy as np, os
import mod_mdt as mod, sys, math
from random import randint
from scipy.special import expit
from copy import deepcopy
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch




class Tracker(): #Tracker
    def __init__(self, parameters):
        self.foldername = parameters.save_foldername + '/0000_CSV'
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.file_save = 'PyTorch_Simulator.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/train_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/valid_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class SSNE_param:
    def __init__(self):
        self.num_input = 21
        self.num_hnodes = 15
        self.num_output = 19

        self.elite_fraction = 0.1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 10000000
        #self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

class Parameters:
    def __init__(self):
            self.population_size = 10
            self.load_seed = False #Loads a seed population from the save_foldername
                                              # IF FALSE: Runs Backpropagation, saves it and uses that
            #BackProp
            self.bprop_max_gens = 1000

            #SSNE stuff
            self.ssne_param = SSNE_param()
            self.total_gens = 100000
            #Determine the nerual archiecture
            self.arch_type = 2 #1 TF_FEEDFORWARD
                               #2 PyTorch GRU-MB


            if self.arch_type == 1: self.arch_type = 'TF_Feedforward'
            elif self.arch_type ==2: self.arch_type = 'PyTorch GRU-MB'
            elif self.arch_type == 3: self.arch_type = 'PyTorch Feedforward'
            else: sys.exit('Invalid choice of neural architecture')

            self.save_foldername = 'PyTorch_Simulator/'

class Fast_Simulator(): #TF Simulator individual (One complete simulator genome)
    def __init__(self):
        self.W = None

    def predict(self, input):
        # Feedforward operation
        h_1 = expit(np.dot(input, self.W[0]) + self.W[1])
        return np.dot(h_1, self.W[2]) + self.W[3]

    def from_tf(self, tf_sess):
        self.W = tf_sess.run(tf.trainable_variables())


class Task_Simulator: #Simulator Task
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output

        self.train_data, self.valid_data = self.data_preprocess() #Get simulator data
        self.ssne = mod.Torch_SSNE(parameters) #nitialize SSNE engine

        # Simulator save folder for checkpoints
        self.marker = 'PT_ANN'
        self.save_foldername = self.parameters.save_foldername
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

        #####CREATE POPULATION
        self.pop = []
        for i in range(self.parameters.population_size):
            self.pop.append(mod.PT_GRUMB(self.num_input, self.num_hidden, self.num_output))

        ###Init population
        if self.parameters.load_seed: #Load seed population
            self.load('bprop_simulator')
        else: #Run Backprop
            self.run_bprop(self.pop[0])

        #mod.simulator_results(self.pop[0])
        #sys.exit()

        # #Init population by randomly perturbing the first one
        # for individual in self.pop[1:]:
        #     individual.W = deepcopy(self.pop[0].W) #Copy pop 0 genome
        #     for w in individual.W:
        #         mut = np.reshape(np.random.normal(0, 2, w.size), (w.shape[0], w.shape[-1]))
        #         w += mut

    def save(self, individual, filename ):
        torch.save(individual, filename)
        #return individual.saver.save(individual.sess, self.save_foldername + filename)

    def load(self, filename):
        return torch.load(self.save_foldername + filename)

    def predict(self, individual, input): #Runs the individual net and computes and output by feedforwarding
        return individual.predict(input)

    def run_bprop(self, model):
        train_x = self.train_data[0:-1]
        train_y = self.train_data[1:,0:-2]
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(self.parameters.bprop_max_gens):
            epoch_loss = 0.0
            for example in range(len(train_x)):  # For all examples
                out = model.forward(train_x[example], is_static = True)

                # Compare with target and compute loss
                y = Variable(torch.Tensor(train_y[example])); y = y.unsqueeze(0)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward(retain_variables=True)
                optimizer.step()
                epoch_loss += loss.data.numpy()[0]
            print 'Epoch: ', epoch, ' Loss: ', epoch_loss / len(train_x)

        self.save(model, self.save_foldername + 'bprop_simulator') #Save individual

    def compute_fitness(self, individual, data):
        fitness = np.zeros(19)
        input = data[0]  # First training example in its entirety
        for example in range(len(data) - 1):  # For all training examples

            model_out = individual.forward(input, is_static=True)# Time domain simulation
            model_out = model_out.data.numpy()
            for index in range(19): # Calculate error (weakness)
                fitness[index] += math.fabs(model_out[0][index] - data[example + 1][index])  # Time variant simulation

            # Fill in new input data
            for k in range(model_out.shape[-1]):
                input[k] = model_out[0][k]
            # Fill in two control variables
            input[19] = data[example + 1][19]
            input[20] = data[example + 1][20]

        return -np.sum(np.square(fitness))/len(data)

    def evolve(self, gen):

        #Fitness evaluation list for the generation
        fitness_evals = [[] for x in xrange(self.parameters.population_size)]

        #Test all individuals and assign fitness
        for index, individual in enumerate(self.pop): #Test all genomes/individuals
            fitness = self.compute_fitness(individual, self.train_data)
            fitness_evals[index] = fitness
        gen_best_fitness = max(fitness_evals)

        #Champion Individual
        champion_index = fitness_evals.index(max(fitness_evals))
        valid_score = self.compute_fitness(self.pop[champion_index], self.valid_data)

        #Save population and HOF
        if gen % 100 == 0:
            for index, individual in enumerate(self.pop): #Save population
                self.save(individual, self.save_foldername + 'Simulator_' + str(index))
            self.save(self.pop[champion_index], self.save_foldername + 'Champion_Simulator') #Save champion
            np.savetxt(self.save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        #SSNE Epoch: Selection and Mutation/Crossover step
        self.ssne.epoch(self.pop, fitness_evals)

        return gen_best_fitness, valid_score

    def data_preprocess(self, filename='ColdAir.csv', downsample_rate=25, split = 1000):
        # Import training data and clear away the two top lines
        data = np.loadtxt(filename, delimiter=',', skiprows=2)

        # Splice data (downsample)
        ignore = np.copy(data)
        data = data[0::downsample_rate]
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (i != data.shape[0] - 1):
                    data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,
                                       j].sum() / downsample_rate
                else:
                    residue = ignore.shape[0] - i * downsample_rate
                    data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue, j].sum() / residue

        # Normalize between 0-0.99
        normalizer = np.zeros(data.shape[1])
        min = np.zeros(len(data[0]))
        max = np.zeros(len(data[0]))
        for i in range(len(data[0])):
            min[i] = np.amin(data[:, i])
            max[i] = np.amax(data[:, i])
            normalizer[i] = max[i] - min[i] + 0.00001
            data[:, i] = (data[:, i] - min[i]) / normalizer[i]

        #Train/Valid split
        train_data = data[0:split]
        valid_data = data[split:len(data)]

        return train_data, valid_data



if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Running Simulator Training ', parameters.arch_type

    sim_task = Task_Simulator(parameters)
    for gen in range(1, parameters.total_gens):
        gen_best_fitness, valid_score = sim_task.evolve(gen)
        print 'Generation:', gen, ' Epoch_reward:', "%0.2f" % gen_best_fitness, ' Valid Score:', "%0.2f" % valid_score, '  Cumul_Valid_Score:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(gen_best_fitness, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(valid_score, gen)  # Add best global performance to tracker














