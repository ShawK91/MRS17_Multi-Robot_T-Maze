import numpy as np, os
import mod_mdt as mod, sys, math
from random import randint
from scipy.special import expit
from torch.autograd import Variable
import torch
import random


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
        self.num_input = 1
        self.num_hnodes = 15
        self.num_output = 1

        self.elite_fraction = 0.1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 10000000
        #self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

class Parameters:
    def __init__(self):
            #BackProp
            self.bprop_max_gens = 50
            self.bprop_train_examples = 100
            self.skip_bprop = False
            self.load_seed = False #Loads a seed population from the save_foldername
                                              # IF FALSE: Runs Backpropagation, saves it and uses that
            #SSNE stuff
            self.population_size = 50
            self.ssne_param = SSNE_param()
            self.total_gens = 100000
            #Determine the nerual archiecture
            self.arch_type = 2 #1 TF_FEEDFORWARD
                               #2 PyTorch GRU-MB

            #Task Params
            self.is_dynamic = True #Makes the task seq len dynamic
            self.dynamic_limit = 50
            self.seq_len_train = 25
            self.seq_len_test = 49
            self.num_train_examples = 25
            self.num_test_examples = 100

            if self.arch_type == 1: self.arch_type = 'TF_Feedforward'
            elif self.arch_type ==2: self.arch_type = 'PyTorch GRU-MB'
            elif self.arch_type == 3: self.arch_type = 'PyTorch Feedforward'
            else: sys.exit('Invalid choice of neural architecture')

            self.save_foldername = 'Binary_Seq_Classifier/'

class Task_Binary_Seq_Classifier: #Bindary Sequence Classifier
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output

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
            self.load('bprop_bsc')
        else: #Run Backprop
            self.run_bprop(self.pop[0])


    def save(self, individual, filename ):
        torch.save(individual, filename)
        #return individual.saver.save(individual.sess, self.save_foldername + filename)

    def load(self, filename):
        return torch.load(self.save_foldername + filename)

    def predict(self, individual, input): #Runs the individual net and computes and output by feedforwarding
        return individual.predict(input)

    def run_bprop(self, model):
        if self.parameters.skip_bprop: return
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_x, train_y = self.test_sequence_data(self.parameters.bprop_train_examples, self.parameters.seq_len_train)

        for epoch in range(1, self.parameters.bprop_max_gens):
            epoch_loss = 0.0
            for example in range(len(train_x)):  # For all examples
                out = model.forward(train_x[example])

                # Compare with target and compute loss
                y = Variable(torch.Tensor(train_y[example])); y = y.unsqueeze(0)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward(retain_variables=True)
                optimizer.step()
                epoch_loss += loss.data.numpy()[0]
            print 'Epoch: ', epoch, ' Loss: ', epoch_loss / len(train_x)



        self.save(model, self.save_foldername + 'bprop_simulator') #Save individual

    def compute_fitness(self, individual, data_x, data_y):
        error = 0.0
        for example in range(len(data_x)):
            out = individual.forward(data_x[example])

            # Compare with target and compute loss
            y = Variable(torch.Tensor(data_y[example])); y = y.unsqueeze(0)
            y_scalar = y.data.numpy()[0][0]
            out_scalar = out.data.numpy()[0][0]
            error = abs(y_scalar - out_scalar)
        return -error/len(data_x)


    def evolve(self, gen):

        #Fitness evaluation list for the generation
        fitness_evals = [[] for x in xrange(self.parameters.population_size)]

        #Get task training examples for the epoch
        train_x, train_y = self.test_sequence_data(self.parameters.num_train_examples, self.parameters.seq_len_train)

        #Test all individuals and assign fitness
        for index, individual in enumerate(self.pop): #Test all genomes/individuals
            fitness = self.compute_fitness(individual, train_x, train_y)
            fitness_evals[index] = fitness
        gen_best_fitness = max(fitness_evals)

        #Champion Individual
        champion_index = fitness_evals.index(max(fitness_evals))
        test_x, test_y = self.test_sequence_data(self.parameters.num_test_examples, self.parameters.seq_len_test)
        valid_score = self.compute_fitness(self.pop[champion_index], test_x, test_y)

        #Save population and HOF
        if gen % 100 == 0:
            for index, individual in enumerate(self.pop): #Save population
                self.save(individual, self.save_foldername + 'Simulator_' + str(index))
            self.save(self.pop[champion_index], self.save_foldername + 'Champion_Simulator') #Save champion
            np.savetxt(self.save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        #SSNE Epoch: Selection and Mutation/Crossover step
        self.ssne.epoch(self.pop, fitness_evals)

        return gen_best_fitness, valid_score

    def test_sequence_data(self, num_examples, seq_len):

        train_x = []; train_y = []
        for example in range(num_examples):
            x = []
            if self.parameters.is_dynamic: #Adjust sequence length dynamically
                seq_len = randint(21, self.parameters.dynamic_limit)
            for pos in range(seq_len):
                if random.random() < 0.5:
                    x.append(1)
                else:
                    x.append(-1)

            if sum(x) < 0:
                train_y.append([0])
            else:
                train_y.append([1])
            train_x.append(x)
        return train_x, train_y



if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Running Simulator Training ', parameters.arch_type

    sim_task = Task_Binary_Seq_Classifier(parameters)
    for gen in range(1, parameters.total_gens):
        gen_best_fitness, valid_score = sim_task.evolve(gen)
        print 'Generation:', gen, ' Epoch_reward:', "%0.2f" % gen_best_fitness, ' Valid Score:', "%0.2f" % valid_score, '  Cumul_Valid_Score:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(gen_best_fitness, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(valid_score, gen)  # Add best global performance to tracker














