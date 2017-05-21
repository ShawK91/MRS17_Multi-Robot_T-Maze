from random import randint
import math
import  cPickle
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import random
import numpy as np, sys, torch
from copy import deepcopy

class PT_GRUMB(nn.Module):
    def __init__(self, input_size, memory_size, output_size):
        super(PT_GRUMB, self).__init__()

        self.input_size = input_size; self.memory_size = memory_size; self.output_size = output_size
        # #Bias placeholders
        # self.input_bias = Variable(torch.ones(1, 1), requires_grad=True)
        # self.rec_input_bias = Variable(torch.ones(1, 1), requires_grad=True)
        # self.mem_bias = Variable(torch.ones(1, 1), requires_grad=True)

        #Input gate
        self.w_inpgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand(output_size, memory_size), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Block Input
        self.w_inp = Parameter(torch.ones(input_size, memory_size), requires_grad=1)
        self.w_rec_inp = Parameter(torch.ones(output_size, memory_size), requires_grad=1)

        #Read Gate
        self.w_readgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(output_size, memory_size), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Write Gate
        self.w_writegate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(output_size, memory_size), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Output weights
        self.w_hid_out = Parameter(torch.rand(memory_size, output_size), requires_grad=1)

        # Adaptive components
        self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        self.out = Variable(torch.zeros(1, self.output_size), requires_grad=0)


    def reset(self):
        # Adaptive components
        self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        self.out = Variable(torch.zeros(1, self.output_size), requires_grad=0)

    def graph_compute(self, input, mem, rec_output):
        # Compute hidden activation
        hidden_act = torch.nn.functional.sigmoid(input.mm(self.w_readgate) + rec_output.mm(self.w_rec_readgate) + mem.mm(self.w_mem_readgate)) * mem + torch.nn.functional.sigmoid(input.mm(self.w_inpgate) + mem.mm(self.w_mem_inpgate) + rec_output.mm(self.w_rec_inpgate)) * torch.nn.functional.sigmoid(input.mm(self.w_inp) + rec_output.mm(self.w_rec_inp))

        #Update mem
        mem = mem + torch.nn.functional.sigmoid(input.mm(self.w_writegate) + mem.mm(self.w_mem_writegate) + rec_output.mm(self.w_rec_writegate))

        # Compute final output
        output = hidden_act.mm(self.w_hid_out)

        return output, mem


    def forward(self, input, is_static=False):
        self.reset()

        if is_static: #Input is one dimensional (non time series data)
            x = Variable(torch.Tensor(input), requires_grad=True); x = x.unsqueeze(0)
            self.out, self.mem = self.graph_compute(x, self.mem, self.out)
        else:
            for item in input:
                x = Variable(torch.Tensor([item]), requires_grad=True); x = x.unsqueeze(0)
                self.out, self.mem = self.graph_compute(x, self.mem, self.out)

        return self.out


    def predict(self, input, is_static=False):
        out = self.forward(input, is_static)
        output = out.data.numpy()
        return output


class FAST_SSNE:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param; self.arch_type = self.parameters.arch_type
        self.population_size = self.parameters.population_size;
        self.num_elitists = int(self.ssne_param.elite_fraction * parameters.population_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output


    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def crossover_inplace(self, gene1, gene2):

        #New weights initialize as copy of previous weights
        new_W1 = gene1.W
        new_W2 = gene2.W


        #Crossover opertation (NOTE THE INDICES CROSSOVER BY COLUMN NOT ROWS)
        num_cross_overs = randint(1, len(new_W1) * 2) #Lower bounded on full swaps
        for i in range(num_cross_overs):
            tensor_choice = randint(0, len(new_W1)-1) #Choose which tensor to perturb
            receiver_choice = random.random() #Choose which gene to receive the perturbation
            if receiver_choice < 0.5:
                ind_cr = randint(0, new_W1[tensor_choice].shape[-1]-1)  #
                new_W1[tensor_choice][:, ind_cr] = new_W2[tensor_choice][:, ind_cr]
            else:
                ind_cr = randint(0, new_W2[tensor_choice].shape[-1]-1)  #
                new_W2[tensor_choice][:, ind_cr] = new_W1[tensor_choice][:, ind_cr]




    def regularize_weight(self, weight):
        if weight > self.ssne_param.weight_magnitude_limit:
            weight = self.ssne_param.weight_magnitude_limit
        if weight < -self.ssne_param.weight_magnitude_limit:
            weight = -self.ssne_param.weight_magnitude_limit
        return weight

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05



        # New weights initialize as copy of previous weights
        new_W = gene.W
        num_variables = len(new_W)


        num_tensor_mutation = randint(0, num_variables-1) #Number of mutation operation level of tensors
        for _ in range(num_tensor_mutation):
            tensor_choice = randint(0, num_variables-1)#Choose which tensor to perturb
            num_mutations = randint(1, math.ceil(num_mutation_frac * new_W[tensor_choice].size)) #Number of mutation instances
            for _ in range(num_mutations):
                ind_dim1 = randint(0, randint(0, new_W[tensor_choice].shape[0]-1))
                ind_dim2 = randint(0, randint(0, new_W[tensor_choice].shape[-1]-1))
                if random.random() < super_mut_prob:
                    new_W[tensor_choice][ind_dim1][ind_dim2] += random.gauss(0, super_mut_strength * new_W[tensor_choice][ind_dim1][ind_dim2])
                else:
                    new_W[tensor_choice][ind_dim1][ind_dim2] += random.gauss(0, mut_strength * new_W[tensor_choice][ind_dim1][ind_dim2])

                # Regularization hard limit
                    new_W[tensor_choice][ind_dim1][ind_dim2] = self.regularize_weight(new_W[tensor_choice][ind_dim1][ind_dim2])

    def copy_individual(self, master, replacee): #Replace the replacee individual with master
       replacee.W = deepcopy(master.W)



    def epoch(self, pop, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)
        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=pop[i], replacee=pop[replacee])
            #pop[replacee] = copy.deepcopy(pop[i])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[randint(0, len(unselects) - 1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            #pop[i] = copy.deepcopy(pop[off_i])
            #pop[j] = copy.deepcopy(pop[off_j])
            self.copy_individual(master=pop[off_i], replacee=pop[i])
            self.copy_individual(master=pop[off_j], replacee=pop[j])
            self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.ssne_param.crossover_prob: self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.ssne_param.mutation_prob:
                    self.mutate_inplace(pop[i])

    def save_pop(self, filename='Pop'):
        filename = str(self.current_gen) + '_' + filename
        pickle_object(self.pop, filename)

class Torch_SSNE:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters;
        self.ssne_param = self.parameters.ssne_param;
        self.arch_type = self.parameters.arch_type
        self.population_size = self.parameters.population_size;
        self.num_elitists = int(self.ssne_param.elite_fraction * parameters.population_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_input = self.ssne_param.num_input;
        self.num_hidden = self.ssne_param.num_hnodes;
        self.num_output = self.ssne_param.num_output

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight):
        if weight > self.ssne_param.weight_magnitude_limit:
            weight = self.ssne_param.weight_magnitude_limit
        if weight < -self.ssne_param.weight_magnitude_limit:
            weight = -self.ssne_param.weight_magnitude_limit
        return weight

    def crossover_inplace(self, gene1, gene2):
        # References to the variable tensors
        W1 = list(gene1.parameters())
        W2 = list(gene2.parameters())
        num_variables = len(W1)
        if num_variables != len(W2): print 'Warning: Genes for crossover might be incompatible'

        # Crossover opertation (NOTE THE INDICES CROSSOVER BY COLUMN NOT ROWS)
        num_cross_overs = randint(1, num_variables * 2)  # Lower bounded on full swaps
        for i in range(num_cross_overs):
            tensor_choice = randint(0, num_variables - 1)  # Choose which tensor to perturb
            receiver_choice = random.random()  # Choose which gene to receive the perturbation
            if receiver_choice < 0.5:
                ind_cr = randint(0, W1[tensor_choice].size()[-1] - 1)  #
                W1[tensor_choice].data[:, ind_cr] = W2[tensor_choice].data[:, ind_cr]
            else:
                ind_cr = randint(0, W2[tensor_choice].size()[-1] - 1)  #
                W2[tensor_choice].data[:, ind_cr] = W1[tensor_choice].data[:, ind_cr]

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        # References to the variable tensors
        W = list(gene.parameters())
        num_variables = len(W)

        num_tensor_mutation = randint(0, num_variables - 1)  # Number of mutation operation level of tensors
        for _ in range(num_tensor_mutation):
            tensor_choice = randint(0, num_variables - 1)  # Choose which tensor to perturb
            num_mutations = randint(1, math.ceil(
                num_mutation_frac * W[tensor_choice].size()[0] * W[tensor_choice].size()[
                    1]))  # Number of mutation instances

            for _ in range(num_mutations):
                ind_dim1 = randint(0, randint(0, W[tensor_choice].size()[0] - 1))
                ind_dim2 = randint(0, randint(0, W[tensor_choice].size()[-1] - 1))
                random_num = random.random()

                if random_num < super_mut_prob: #Super Mutation probability
                    W[tensor_choice].data[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                              W[tensor_choice].data[ind_dim1, ind_dim2])
                elif random_num < reset_prob: #Reset probability
                    W[tensor_choice].data[ind_dim1, ind_dim2] = random.gauss(0, W[tensor_choice].data[ind_dim1, ind_dim2])

                else: #mutauion even normal
                    W[tensor_choice].data[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * W[tensor_choice].data[
                        ind_dim1, ind_dim2])

                    # Regularization hard limit
                    W[tensor_choice].data[ind_dim1, ind_dim2] = self.regularize_weight(W[tensor_choice].data[ind_dim1, ind_dim2])

    def copy_individual(self, master, replacee):  # Replace the replacee individual with master

        W_master = list(master.parameters())
        W_replacee = list(replacee.parameters())
        for w_replacee, w_master in zip(W_replacee, W_master):
            w_replacee.data[:] = w_master.data

    def reset_genome(self, gene):
        W = list(gene.parameters())
        for tensor in W:
            dim1 = tensor.size()[0]
            dim2 = tensor.size()[-1]
            tensor.data = torch.rand(dim1, dim2)

    def epoch(self, pop, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals);
        index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        #Extinction step (Resets all the offsprings genes; preserves the elitists)
        if random.random() < self.ssne_param.extinction_prob: #An extinction event
            print "######################Extinction Event Triggered#######################"
            for i in offsprings:
                if random.random() < self.ssne_param.extinction_magnituide and not (i in elitist_index):  # Extinction probabilities
                    self.reset_genome(pop[i])
        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=pop[i], replacee=pop[replacee])
            # pop[replacee] = copy.deepcopy(pop[i])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[randint(0, len(unselects) - 1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            self.copy_individual(master=pop[off_i], replacee=pop[i])
            self.copy_individual(master=pop[off_j], replacee=pop[j])
            self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.ssne_param.crossover_prob: self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.ssne_param.mutation_prob:
                    self.mutate_inplace(pop[i])

    def save_model(self, model, filename):
        torch.save(model, filename)

class TF_SSNE:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param; self.arch_type = self.parameters.arch_type
        self.population_size = self.parameters.population_size;
        self.num_elitists = int(self.ssne_param.elite_fraction * parameters.population_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output


    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def crossover_inplace(self, gene1, gene2):
        #References to the variable tensors
        variable_tensors= tf.trainable_variables()
        num_variables = len(variable_tensors)

        #New weights initialize as copy of previous weights
        new_W1 = gene1.sess.run(tf.trainable_variables())
        new_W2 = gene2.sess.run(tf.trainable_variables())


        #Crossover opertation (NOTE THE INDICES CROSSOVER BY COLUMN NOT ROWS)
        num_cross_overs = randint(1, num_variables * 2) #Lower bounded on full swaps
        for i in range(num_cross_overs):
            tensor_choice = randint(0, num_variables-1) #Choose which tensor to perturb
            receiver_choice = random.random() #Choose which gene to receive the perturbation
            if receiver_choice < 0.5:
                ind_cr = randint(0, new_W1[tensor_choice].shape[-1]-1)  #
                new_W1[tensor_choice][:, ind_cr] = new_W2[tensor_choice][:, ind_cr]
            else:
                ind_cr = randint(0, new_W2[tensor_choice].shape[-1]-1)  #
                new_W2[tensor_choice][:, ind_cr] = new_W1[tensor_choice][:, ind_cr]

        #Assign the new weights to individuals
        for i in range(num_variables):
            #Create operations for assigning new weights
            op_1 = variable_tensors[i].assign(new_W1[i])
            op_2 = variable_tensors[i].assign(new_W2[i])

            #Run them in session
            gene1.sess.run(op_1)
            gene2.sess.run(op_2)

    def regularize_weight(self, weight):
        if weight > self.ssne_param.weight_magnitude_limit:
            weight = self.ssne_param.weight_magnitude_limit
        if weight < -self.ssne_param.weight_magnitude_limit:
            weight = -self.ssne_param.weight_magnitude_limit
        return weight

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05

        # References to the variable tensors
        variable_tensors = tf.trainable_variables()
        num_variables = len(variable_tensors)

        # New weights initialize as copy of previous weights
        new_W = gene.sess.run(tf.trainable_variables())


        num_tensor_mutation = randint(0, num_variables-1) #Number of mutation operation level of tensors
        for _ in range(num_tensor_mutation):
            tensor_choice = randint(0, num_variables-1)#Choose which tensor to perturb
            num_mutations = randint(1, math.ceil(num_mutation_frac * new_W[tensor_choice].size)) #Number of mutation instances
            for _ in range(num_mutations):
                ind_dim1 = randint(0, randint(0, new_W[tensor_choice].shape[0]-1))
                ind_dim2 = randint(0, randint(0, new_W[tensor_choice].shape[-1]-1))
                if random.random() < super_mut_prob:
                    new_W[tensor_choice][ind_dim1][ind_dim2] += random.gauss(0, super_mut_strength * new_W[tensor_choice][ind_dim1][ind_dim2])
                else:
                    new_W[tensor_choice][ind_dim1][ind_dim2] += random.gauss(0, mut_strength * new_W[tensor_choice][ind_dim1][ind_dim2])

                # Regularization hard limit
                    new_W[tensor_choice][ind_dim1][ind_dim2] = self.regularize_weight(new_W[tensor_choice][ind_dim1][ind_dim2])

        # Assign the new weights to individuals
        for i in range(num_variables):
            # Create operations for assigning new weights
            op_1 = variable_tensors[i].assign(new_W[i])

            # Run them in session
            gene.sess.run(op_1)

    def copy_individual(self, master, replacee): #Replace the replacee individual with master
        #References to the variable tensors
        variable_tensors= tf.trainable_variables()
        num_variables = len(variable_tensors)

        #New weights initialize as copy of previous weights
        master_W = master.sess.run(tf.trainable_variables())

        #Assign the new weights to individuals
        for i in range(num_variables):
            #Create operations for assigning new weights
            op = variable_tensors[i].assign(master_W[i])

            #Run them in session
            replacee.sess.run(op)



    def epoch(self, pop, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)
        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=pop[i], replacee=pop[replacee])
            #pop[replacee] = copy.deepcopy(pop[i])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[randint(0, len(unselects) - 1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            #pop[i] = copy.deepcopy(pop[off_i])
            #pop[j] = copy.deepcopy(pop[off_j])
            self.copy_individual(master=pop[off_i], replacee=pop[i])
            self.copy_individual(master=pop[off_j], replacee=pop[j])
            self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.ssne_param.crossover_prob: self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.ssne_param.mutation_prob:
                    self.mutate_inplace(pop[i])

    def save_pop(self, filename='Pop'):
        filename = str(self.current_gen) + '_' + filename
        pickle_object(self.pop, filename)

def simulator_results(model, filename='ColdAir.csv', downsample_rate=25):
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

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

    print ('TESTING NOW')
    input = data[0]  # First input to the simulatior
    track_target = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))
    track_output = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))

    for example in range(len(data) - 1):  # For all training examples
        model_out = model.predict(input, is_static=True)

        # Track index
        for index in range(19):
            track_output[index][example] = model_out[0][index]  # * normalizer[index] + min[index]
            track_target[index][example] = data[example + 1][index]  # * normalizer[index] + min[index]

        # Fill in new input data
        for k in range(len(model_out[0])):
            input[k] = model_out[k]
        # Fill in two control variables
        input[19] = data[example + 1][19]
        input[20] = data[example + 1][20]

    for index in range(19):
        plt.plot(track_target[index], 'r--', label='Actual Data: ' + str(index))
        plt.plot(track_output[index], 'b-', label='TF_Simulator: ' + str(index))
        # np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
        # np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
        plt.legend(loc='upper right', prop={'size': 6})
        # plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
        # print track_output[index]
        plt.show()







#BACKUPS
def pstats():
    import pstats
    p = pstats.Stats('profile.profile')
    p.sort_stats('cumulative').print_stats(10)
    p.sort_stats('cumulative').print_stats(50)
    p.sort_stats('cumulative').print_stats(50)

def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)

def return_mem_address(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]