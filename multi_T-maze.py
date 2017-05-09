import numpy as np, os
import mod_mdt as mod, sys
from random import randint
from torch.autograd import Variable
import torch.nn as nn, torch
from torch.nn import Parameter
import torch
import random

class Tracker(): #Tracker
    def __init__(self, parameters):
        self.foldername = parameters.save_foldername + '/0000_CSV'
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.file_save = 'PyTorch_Multi_TMaze.csv'

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
        self.num_input = 3
        self.num_hnodes = 15
        self.num_output = 1

        self.elite_fraction = 0.1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 10000000
        #self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

class Parameters:
    def __init__(self):

            #SSNE stuff
            self.population_size = 100
            self.ssne_param = SSNE_param()
            self.total_gens = 100000
            #Determine the nerual archiecture
            self.arch_type = 2 #1 TF_FEEDFORWARD
                               #2 PyTorch GRU-MB

            #Task Params
            self.depth = 2
            self.is_dynamic_depth = False #Makes the task seq len dynamic
            self.dynamic_depth_limit = [1,10]

            self.corridor_bound = [1,1]
            self.num_evals_ccea = 5 #Number of different teams to test the same individual in before assigning a score

            self.num_trials = pow(2, self.depth) * 3 #One trial is the robot going to a goal location. One evaluation consistis to multiple trials


            #Multi-Agent Params
            self.num_agents = 2

            #Reward
            self.rew_multi_success = 1.0
            self.rew_single_success = 0.2
            self.rew_same_path = 0.05


            if self.arch_type == 1: self.arch_type = 'TF_Feedforward'
            elif self.arch_type ==2: self.arch_type = 'PyTorch GRU-MB'
            elif self.arch_type == 3: self.arch_type = 'PyTorch Feedforward'
            else: sys.exit('Invalid choice of neural architecture')

            self.save_foldername = 'R_Multi-Agent_TMaze/'

            # BackProp
            self.bprop_max_gens = 100
            self.bprop_train_examples = 1000
            self.skip_bprop = False
            self.load_seed = False  # Loads a seed population from the save_foldername
            # IF FALSE: Runs Backpropagation, saves it and uses that


class GRUMB(nn.Module):
    def __init__(self, input_size, memory_size, output_size):
        super(GRUMB, self).__init__()

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

        #Adaptive components
        self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        self.out = Variable(torch.zeros(1, self.output_size), requires_grad=0)
        self.agent_sensor = 0.0; self.last_reward = 0.0

    def reset(self):
        #Adaptive components
        self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        self.out = Variable(torch.zeros(1, self.output_size), requires_grad=0)
        self.agent_sensor = 0.0; self.last_reward = 0.0

    def graph_compute(self, input, mem, rec_output):
        # Compute hidden activation
        hidden_act = torch.nn.functional.sigmoid(input.mm(self.w_readgate) + rec_output.mm(self.w_rec_readgate) + mem.mm(self.w_mem_readgate)) * mem + torch.nn.functional.sigmoid(input.mm(self.w_inpgate) + mem.mm(self.w_mem_inpgate) + rec_output.mm(self.w_rec_inpgate)) * torch.nn.functional.sigmoid(input.mm(self.w_inp) + rec_output.mm(self.w_rec_inp))

        #Update mem
        mem = mem + torch.nn.functional.sigmoid(input.mm(self.w_writegate) + mem.mm(self.w_mem_writegate) + rec_output.mm(self.w_rec_writegate))

        # Compute final output
        output = hidden_act.mm(self.w_hid_out)

        return output, mem

    def forward(self, input):
        self.out, self.mem = self.graph_compute(input, self.mem, self.out)

        return self.out

    def predict(self, input, is_static=False):
        out = self.forward(input, is_static)
        output = out.data.numpy()
        return output


class Agent_Pop:
    def __init__(self, parameters, i):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output
        self.agent_id = i

        #####CREATE POPULATION
        self.pop = []
        for _ in range(self.parameters.population_size):
            self.pop.append(GRUMB(self.num_input, self.num_hidden, self.num_output))

        #Fitness evaluation list for the generation
        self.fitness_evals = [[0.0] for x in xrange(self.parameters.population_size)]
        self.selection_pool = [i for i in range(self.parameters.population_size)]*self.parameters.num_evals_ccea


    def reset(self):
        #Fitness evaluation list for the generation
        self.fitness_evals = [[0.0] for x in xrange(self.parameters.population_size)]
        self.selection_pool = [i for i in range(self.parameters.population_size)] * self.parameters.num_evals_ccea


class Task_Multi_TMaze: #Mulit-Agent T-Maze
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output

        self.ssne = mod.Torch_SSNE(parameters) #nitialize SSNE engine

        # Simulator save folder for checkpoints
        self.marker = 'PT_ANN'
        self.save_foldername = self.parameters.save_foldername
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

        #####Initiate all agents
        self.all_agents = []
        for agent_id in range(self.parameters.num_agents):
            self.all_agents.append(Agent_Pop(parameters, agent_id))

        # ###Init population
        # if self.parameters.load_seed: #Load seed population
        #     self.load('bprop_bsc')
        # else: #Run Backprop
        #     self.run_bprop(self.pop[0])

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

    def compute_fitness(self, team_ind, train_x, train_y):

        team = [] #Grab reference to the individual agents themeselves for ease of use from team_index
        #Reset memory and out for all agents
        for i, agent_index  in enumerate(team_ind):
            team.append(self.all_agents[i].pop[agent_index])
            team[-1].reset()


        fitnesses = np.zeros(len(team)) #Stores the actual fitnesses
        for trial in range(self.parameters.num_trials): #For each trial

            #Encode last is_another agent and reward value from last time and forward propagate it
            for individual in team:
                x = Variable(torch.Tensor([0, individual.agent_sensor, individual.last_reward]), requires_grad=True); x = x.unsqueeze(0)
                individual.forward(x)

            #Reset path out that collects path out for each individual
            path_out = []
            for _ in range(len(team)): path_out.append([])

            #Run through the maze and collect path chosen by each individual
            for i, individual in enumerate(team):
                for step in train_x:
                    x = Variable(torch.Tensor([step, 0, 0]), requires_grad=True); x = x.unsqueeze(0)
                    out = individual.forward(x)
                    if step == 0: #For each junction
                        net_out = out.data.numpy()[0]
                        if net_out < 0: path_out[i].append(-1)
                        else: path_out[i].append(1)


            y = train_y[trial/(self.parameters.num_trials/3)]  # Pick the correct reward location
            #Update reward and is_other agent
            is_multi_success = True #Did all agents converge at reward?
            for i, individual in enumerate(team):
                individual.last_reward = 0.0;  individual.agent_sensor = 0.0 #Reset last reward value and agent sensor
                if path_out[i] == y: individual.last_reward += self.parameters.rew_single_success #Single agent reward
                else: is_multi_success = False

            #Did all agents take the same path?
            is_same_path = True
            for path1 in path_out:
                for path2 in path_out:
                    if path1 != path2:
                        is_same_path = False
                        break

                if not is_same_path: break;

            #Give reward for multi_agent_success
            if is_multi_success:
                for individual in team: individual.last_reward += self.parameters.rew_multi_success
            if is_same_path:
                for individual in team:
                    individual.last_reward += self.parameters.rew_same_path
                    individual.agent_sensor = 1.0


            #Disburse fitnesses
            for i, individual in enumerate(team):
                fitnesses[i] += individual.last_reward

        return fitnesses/(self.parameters.num_trials*len(team_ind)) #


    def evolve(self, gen):
        best_epoch_fitness = 0.0

        # Reset all agent's fitness and sampling pool
        for agent_pop in self.all_agents:
            agent_pop.reset()
        team_ind = np.zeros(self.parameters.num_agents).astype(int)  # Team definitions by index

        #Generate training map and solution to test on for the epoch
        train_x, train_y = self.get_training_maze()

        # MAIN LOOP
        for _ in range(self.parameters.population_size * self.parameters.num_evals_ccea):  # For evaluation

            # PICK TEAMS
            for i, agent_pop in enumerate(self.all_agents):
                choice = randint(0, len(agent_pop.selection_pool)-1)
                team_ind[i] = agent_pop.selection_pool.pop(choice)

            # SIMULATION AND TRACK REWARD
            fitnesses = self.compute_fitness(team_ind, train_x, train_y)  # Returns rewards for each member of the team
            if sum(fitnesses) > best_epoch_fitness: best_epoch_fitness = sum(fitnesses)

            # ENCODE fitness back to each agent populations
            for i, agent_pop in enumerate(self.all_agents):
                agent_pop.fitness_evals[team_ind[i]] += fitnesses[i]


        # #Save population and HOF
        # if gen % 100 == 0:
        #     for index, individual in enumerate(self.pop): #Save population
        #         self.save(individual, self.save_foldername + 'Simulator_' + str(index))
        #     self.save(self.pop[champion_index], self.save_foldername + 'Champion_Simulator') #Save champion
        #     np.savetxt(self.save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        #SSNE Epoch: Selection and Mutation/Crossover step
        for agent_pop in self.all_agents:
            #Obtain fitness evals as an average of the fitness evaluations fone during the epoch
            fitness_evals = []
            for samples in agent_pop.fitness_evals: fitness_evals.append(sum(samples)/len(samples))

            self.ssne.epoch(agent_pop.pop, fitness_evals)

        return best_epoch_fitness

    def get_training_maze(self):
        # Distraction/Hallway parts
        train_x = []; train_y = [[],[],[]]
        for junction in range(self.parameters.depth):
            corridor_len = randint(self.parameters.corridor_bound[0], self.parameters.corridor_bound[1])
            for i in range(corridor_len-1, -1, -1):
                train_x.append(i)
            for div in train_y:
                div.append(1 if random.random()<0.5 else -1)
        return train_x, train_y



if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Running Simulator Training ', parameters.arch_type

    sim_task = Task_Multi_TMaze(parameters)
    for gen in range(1, parameters.total_gens):
        gen_best_fitness = sim_task.evolve(gen)
        print 'Generation:', gen, ' Epoch_reward:', "%0.2f" % gen_best_fitness#, ' Valid Score:', "%0.2f" % valid_score, '  Cumul_Valid_Score:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(gen_best_fitness, gen)  # Add average global performance to tracker
        #tracker.add_hof_fitness(valid_score, gen)  # Add best global performance to tracker














