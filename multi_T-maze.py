import numpy as np, os
import mod_mdt as mod, sys
from random import randint
from torch.autograd import Variable
import torch.nn as nn, torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import random
from operator import add

#TODO Weight initialization Torch
#TODO Compute fitness function optimize



is_probe= False

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
            filename = self.foldername + '/champ_train' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/champ_real' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class SSNE_param:
    def __init__(self):
        self.num_input = 3
        self.num_hnodes = 20
        self.num_output = 1

        self.elite_fraction = 0.03
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.extinction_prob = 0.004 #Probability of extinction event
        self.extinction_magnituide = 0.9 #Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 1000000
        #self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

class Parameters:
    def __init__(self):

        #SSNE stuff
        self.population_size = 100
        self.load_seed = False
        self.ssne_param = SSNE_param()
        self.total_gens = 100000
        #Determine the nerual archiecture
        self.arch_type = 2 #1 LSTM
                           #2 GRUMB
                           #3 FF

        #Task Params
        self.depth = 2
        self.is_dynamic_depth = False #Makes the task seq len dynamic
        self.dynamic_depth_limit = [1,10]

        self.corridor_bound = [1,1]
        self.num_evals_ccea = 1 #Number of different teams to test the same individual in before assigning a score
        self.num_train_evals = 10 #Number of different maps to run each individual before getting a fitness

        self.num_trials = pow(2, self.depth) * 3 #One trial is the robot going to a goal location. One evaluation consistis to multiple trials


        #Multi-Agent Params
        self.static_policy = True #Agent 0 is static policy (num_agents includes this)
        self.is_static_variable = False #Determines if the static policy is variable
        self.num_agents = 2

        #Reward (Real fitness will always measure multi_success only)
        self.rew_multi_success = 1.0  /(self.num_trials)
        self.rew_single_success = 0.0  /(self.num_trials)
        self.rew_same_path = 0.0  /(self.num_trials)
        self.explore_reward = 0.0  /(self.num_trials-1)
        self.pure_exploration = False



        self.output_activation = None
        if self.arch_type == 1: self.arch_type = 'LSTM'
        elif self.arch_type ==2: self.arch_type = 'GRUMB'
        elif self.arch_type == 3: self.arch_type = 'FF'
        else: sys.exit('Invalid choice of neural architecture')

        self.save_foldername = 'R_Multi-Agent_TMaze/'

        # # BackProp
        # self.bprop_max_gens = 100
        # self.bprop_train_examples = 1000
        # self.skip_bprop = False
        # self.load_bprop_seed = False  # Loads a seed population from the save_foldername
        # # IF FALSE: Runs Backpropagation, saves it and uses that

        print 'Depth:', self.depth, '  Num_Trials:', self.num_trials, '  Num_Agents:', self.num_agents, ' Is_Static:', self.static_policy, ' Exploration only:', self.pure_exploration

        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

        #Overrides
        if is_probe: self.load_seed = True #Overrides



class LSTM(nn.Module):
    def __init__(self,input_size, memory_size, output_size):
        super(LSTM, self).__init__()
        self.is_static = False #Distinguish between this and static policy
        self.input_size = input_size; self.memory_size = memory_size; self.output_size = output_size
        #self.lstm= nn.LSTM(input_size, memory_size, output_size)

        #LSTM implementation
        # Input gate
        self.w_inpgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Block Input
        self.w_inp = Parameter(torch.ones(input_size, memory_size), requires_grad=1)
        self.w_rec_inp = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)

        # Forget gate
        self.w_forgetgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_forgetgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)
        self.w_mem_forgetgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Output gate
        self.w_outgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_outgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)
        self.w_mem_outgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Hidden_to_out
        self.hid_out = Parameter(torch.rand(memory_size, output_size), requires_grad=1)


        #Adaptive components
        self.c = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        self.h = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        self.agent_sensor = 0.0; self.last_reward = 0.0

        # Turn grad off for evolutionary algorithm
        self.turn_grad_off()

    def reset(self):
        #Adaptive components
        self.c = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        self.h = Variable(torch.zeros(1, self.memory_size), requires_grad=0)

        self.agent_sensor = 0.0; self.last_reward = 0.0

    def graph_compute(self, input):
        inp_gate = F.sigmoid(input.mm(self.w_inpgate) + self.h.mm(self.w_rec_inpgate) + self.c.mm(self.w_mem_inpgate))
        forget_gate = F.sigmoid(input.mm(self.w_forgetgate) + self.h.mm(self.w_rec_forgetgate) + self.c.mm(self.w_mem_forgetgate))
        out_gate = F.sigmoid(input.mm(self.w_outgate) + self.h.mm(self.w_rec_outgate) + self.c.mm(self.w_mem_outgate))

        ct_new = F.tanh(input.mm(self.w_inp) + self.h.mm(self.w_rec_inp))

        c_t = inp_gate * ct_new + forget_gate * self.c
        h_t = out_gate * F.tanh(c_t)
        return h_t, c_t

    def forward(self, input):
        self.h, self.c = self.graph_compute(input)

        return self.hid_out.mm(self.h)

    def predict(self, input, is_static=False):
        out = self.forward(input, is_static)
        output = out.data.numpy()
        return output

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True

#
# class GRUMB(nn.Module):
#     def __init__(self, input_size, memory_size, output_size):
#         super(GRUMB, self).__init__()
#         self.is_static = False  # Distinguish between this and static policy
#
#         self.input_size = input_size;
#         self.memory_size = memory_size;
#         self.output_size = output_size
#         # #Bias placeholders
#         # self.input_bias = Variable(torch.ones(1, 1), requires_grad=True)
#         # self.rec_input_bias = Variable(torch.ones(1, 1), requires_grad=True)
#         # self.mem_bias = Variable(torch.ones(1, 1), requires_grad=True)
#
#         # Input gate
#         self.w_inpgate = Parameter(torch.rand(input_size, memory_size), requires_grad=0)
#         self.w_rec_inpgate = Parameter(torch.rand(output_size, memory_size), requires_grad=0)
#         self.w_mem_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=0)
#
#         # Block Input
#         self.w_inp = Parameter(torch.ones(input_size, memory_size), requires_grad=0)
#         self.w_rec_inp = Parameter(torch.ones(output_size, memory_size), requires_grad=0)
#
#         # Read Gate
#         self.w_readgate = Parameter(torch.rand(input_size, memory_size), requires_grad=0)
#         self.w_rec_readgate = Parameter(torch.rand(output_size, memory_size), requires_grad=0)
#         self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=0)
#
#         # Write Gate
#         self.w_writegate = Parameter(torch.rand(input_size, memory_size), requires_grad=0)
#         self.w_rec_writegate = Parameter(torch.rand(output_size, memory_size), requires_grad=0)
#         self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=0)
#
#         # Output weights
#         self.w_hid_out = Parameter(torch.rand(memory_size, output_size), requires_grad=0)
#
#         # Adaptive components
#         self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
#         self.out = Variable(torch.zeros(1, self.output_size), requires_grad=0)
#         self.agent_sensor = 0.0;
#         self.last_reward = 0.0
#
#         # Turn grad off for evolutionary algorithm
#         self.turn_grad_off()
#
#     def reset(self):
#         # Adaptive components
#         self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
#         self.out = Variable(torch.zeros(1, self.output_size), requires_grad=0)
#         self.agent_sensor = 0.0;
#         self.last_reward = 0.0
#
#     def graph_compute(self, input, mem, rec_output):
#         # Compute hidden activation
#         hidden_act = torch.nn.functional.sigmoid(
#             input.mm(self.w_readgate) + rec_output.mm(self.w_rec_readgate) + mem.mm(
#                 self.w_mem_readgate)) * mem + torch.nn.functional.sigmoid(
#             input.mm(self.w_inpgate) + mem.mm(self.w_mem_inpgate) + rec_output.mm(
#                 self.w_rec_inpgate)) * torch.nn.functional.sigmoid(input.mm(self.w_inp) + rec_output.mm(self.w_rec_inp))
#
#         # Update mem
#         mem = mem + torch.nn.functional.sigmoid(
#             input.mm(self.w_writegate) + mem.mm(self.w_mem_writegate) + rec_output.mm(self.w_rec_writegate))
#
#         # Compute final output
#         output = hidden_act.mm(self.w_hid_out)
#
#         return output, mem
#
#     def forward(self, input):
#         self.out, self.mem = self.graph_compute(input, self.mem, self.out)
#
#         return self.out
#
#     def predict(self, input, is_static=False):
#         out = self.forward(input, is_static)
#         output = out.data.numpy()
#         return output
#
#     def turn_grad_on(self):
#         for param in self.parameters():
#             param.requires_grad = True
#             param.volatile = False
#
#     def turn_grad_off(self):
#         for param in self.parameters():
#             param.requires_grad = False
#             param.volatile = True

class FF(nn.Module):
    def __init__(self, input_size, memory_size, output_size):
        super(FF, self).__init__()
        self.is_static = False #Distinguish between this and static policy

        self.input_size = input_size; self.memory_size = memory_size; self.output_size = output_size

        #Block Input
        self.w_inp = Parameter(torch.ones(input_size, memory_size), requires_grad=1)

        #Output weights
        self.w_hid_out = Parameter(torch.rand(memory_size, output_size), requires_grad=1)

        #Adaptive components
        self.agent_sensor = 0.0; self.last_reward = 0.0

        # Turn grad off for evolutionary algorithm
        self.turn_grad_off()


    def reset(self):
        #Adaptive components
        self.agent_sensor = 0.0; self.last_reward = 0.0

    def graph_compute(self, input):
        return F.sigmoid(input.mm(self.w_inp)).mm(self.w_hid_out)

    def forward(self, input):
        return self.graph_compute(input)

    def predict(self, input, is_static=False):
        out = self.forward(input, is_static)
        output = out.data.numpy()
        return output

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True

class Static_policy():
    def __init__(self, parameters):
        self.paramters = parameters; self.fast_net = self
        self.is_static = True
        self.out = np.ones(self.paramters.depth)  # PLACEHOLDER
        if self.paramters.is_static_variable:  # if variable static policy
            self.bank_out = self.init_exploration_start_point()  # Randomly initialize where the static policy starts from and store it as bank starting point for this specific policy
            self.explore_policy = self.binary_add if random.random() <0.5 else self.binary_substract
        else: #Static policy is not variable
            self.bank_out = np.ones(self.paramters.depth)
            self.explore_policy = self.binary_add

        #Resettable
        self.junction_id = -1
        self.out[:] = self.bank_out #Initialize out as the bank initialization point
        self.agent_sensor = 0
        self.last_reward = 0.0
        self.new_trial = True

    def reset(self):
        self.junction_id = -1
        self.agent_sensor = -1.0
        self.last_reward = 0.0
        self.out[:] = self.bank_out
        self.new_trial = True

    def init_exploration_start_point(self):
        out = np.zeros(self.paramters.depth)
        for i in range(self.paramters.depth):
            if random.random() < 0.5: out[i] = 1
        return out

    def forward(self, input):
        if input[0] != 0 and not self.new_trial: #Wall and distractors
            return

        if self.new_trial: #First forward propagation (ignore) (Update the self.out for the trial)
            self.new_trial = False
            self.last_reward = input[2]
            self.junction_id = -1

            #Compute output
            if self.last_reward == 1: None #if found reward last time
            else: #Explore policy (could be binary add or binary substract)
                self.explore_policy()

        else: #Decision taking points (junctions)
            self.junction_id += 1
            return [[self.out[self.junction_id]]]
            #return Variable(torch.Tensor([[self.out[self.junction_id]]]), requires_grad=0)


    def binary_add(self):
        carry = 0
        for j in range(len(self.out) - 1, -1, -1):
            if carry != 0:
                self.out[j] += carry
                carry = 0
            if j == len(self.out) - 1: self.out[j] += 1
            if self.out[j] == 2:
                carry = 1
                self.out[j] = 0
        if carry == 1:
            self.out = np.zeros(self.paramters.depth)  # Boundary condition


    def binary_substract(self):
        carry = 0
        for j in range(len(self.out) - 1, -1, -1):
            if carry != 0:
                self.out[j] += carry
                carry = 0
            if j == len(self.out) - 1: self.out[j] -= 1
            if self.out[j] == -1:
                carry = -1
                self.out[j] = 1
        if carry == -1:
            self.out = np.ones(self.paramters.depth)  # Boundary condition

class Agent_Pop:
    def __init__(self, parameters, i, is_static=False):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output
        self.agent_id = i
        self.is_static = is_static

        #####CREATE POPULATION
        self.pop = []
        if is_static: #Static policy agent
            for i in range(self.parameters.population_size):
                self.pop.append(Static_policy(parameters))
        else:
            for i in range(self.parameters.population_size):
                if self.parameters.load_seed and i == 0: #Load seed if option
                    self.pop.append(self.load('champion_' + str(self.agent_id)))

                #Choose architecture
                if self.parameters.arch_type == "GRUMB":
                    self.pop.append(mod.PT_GRUMB(self.num_input, self.num_hidden, self.num_output, output_activation=self.parameters.output_activation))
                elif self.parameters.arch_type == "FF":
                    self.pop.append(FF(self.num_input, self.num_hidden, self.num_output))
                elif self.parameters.arch_type == "LSTM":
                    self.pop.append(LSTM(self.num_input, self.num_hidden, self.num_output))
                else:
                    sys.exit('Invalid choice of architecture')
        self.champion_ind = None

        #Fitness evaluation list for the generation
        self.fitness_evals = [0.0] * self.parameters.population_size
        self.selection_pool = [i for i in range(self.parameters.population_size)]*self.parameters.num_evals_ccea


    def reset(self):
        #Fitness evaluation list for the generation
        self.fitness_evals = [0.0] * self.parameters.population_size
        self.selection_pool = [i for i in range(self.parameters.population_size)]*self.parameters.num_evals_ccea


    def load(self, filename):
        return torch.load(self.parameters.save_foldername + filename)

class Task_Multi_TMaze: #Mulit-Agent T-Maze
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output

        self.ssne = mod.Fast_SSNE(parameters) #nitialize SSNE engine

        #####Initiate all agents
        self.all_agents = []
        for agent_id in range(self.parameters.num_agents):
            if self.parameters.static_policy and agent_id == 0: #Static policy for agent_id 0
                self.all_agents.append(Agent_Pop(parameters, agent_id, is_static=True))
            else:
                self.all_agents.append(Agent_Pop(parameters, agent_id))

        if is_probe: self.run_probe() #Run probe and end the program



    def run_probe(self):
        team_ind = [0]*self.parameters.num_agents #Grab all the loaded champions

        set_train_x, set_train_y = self.get_training_maze(self.parameters.num_train_evals)
        fitnesses = [0] * self.parameters.num_agents
        for train_x, train_y in zip(set_train_x, set_train_y):  # For all maps in the training set
            fitnesses = map(add, fitnesses, self.compute_fitness(team_ind, train_x, train_y) / self.parameters.num_train_evals)  # Returns rewards for each member of the team for each individual training map
        #TODO FINISH PROBE

        sys.exit('PROBE COMPLETED')

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

        team_last_rewards=[0]*len(team_ind); team_last_is_agents = [0] * len(team_ind) #Team last reward and is same path last agent for last trial
        team = [] #Grab reference to the individual agents themeselves for ease of use from team_index
        #Reset memory and out for all agents
        for i, agent_index  in enumerate(team_ind):
            team.append(self.all_agents[i].pop[agent_index])
            team[-1].fast_net.reset()


        all_path_simulation = [[[],[]] for _ in range (self.parameters.num_agents)] #Store all path taken by each agent through the entrie simulation (all trials))
        training_fitnesses = np.zeros(len(team)) #Stores the training fitnesses
        real_fitness = 0.0  # Stores the actual fitnesses
        for trial in range(self.parameters.num_trials): #For each trial

            #Encode last is_another agent and reward value from last time and forward propagate it
            for individual, last_reward, is_agent in zip(team, team_last_rewards, team_last_is_agents):
                try:
                    if individual.is_static: individual.new_trial = True
                except: True
                x = [0,  is_agent, last_reward]
                #x = Variable(torch.Tensor([0,  is_agent, last_reward]), requires_grad=True); x = x.unsqueeze(0)
                individual.fast_net.forward(x)

            #Reset path out that collects path out for each individual for each trial
            path_out = [[] for _ in range(self.parameters.num_agents)]

            #Run through the maze and collect path chosen by each individual
            for i, individual in enumerate(team):
                for step in train_x:
                    x = [step, 0, 0]
                    #x = Variable(torch.Tensor([step, 0, 0]), requires_grad=True); x = x.unsqueeze(0)
                    out = individual.fast_net.forward(x)
                    if step == 0: #For each junction
                        net_out = out[0][0]
                        if net_out <= 0: path_out[i].append(-1)
                        else: path_out[i].append(1)
                all_path_simulation[i][0].append(path_out[i])


            y = train_y[trial/(self.parameters.num_trials/3)]  # Pick the correct reward location
            #Reset reward and is_other agent, and also single rewards
            is_multi_success = True #Did all agents converge at reward?
            for i, individual in enumerate(team):
                team_last_rewards[i] = 0.0;  team_last_is_agents[i] = 0.0 #Reset last reward value and agent sensor
                if path_out[i] == y:
                    training_fitnesses[i] += self.parameters.rew_single_success #Single agent reward
                    team_last_rewards[i] = 1.0
                    all_path_simulation[i][1].append(1)
                else:
                    is_multi_success = False
                    all_path_simulation[i][1].append(0)


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
                for i in range(len(team)):
                    training_fitnesses[i] += self.parameters.rew_multi_success
                real_fitness += 1.0 /(self.parameters.num_trials)
            if is_same_path:
                for i in range(len(team)):
                    training_fitnesses[i] += self.parameters.rew_same_path
                    team_last_is_agents[i] = 1.0


        #Reward based on exploration
        for individual_id, individual in enumerate(team):
            for path_id, single_path in enumerate(all_path_simulation[individual_id][0]):
                if path_id == 0: continue #Skip the first path
                is_explore = True
                for history in range(self.parameters.num_trials/3):
                    check_index = path_id - history -1
                    if check_index < 0: break; #Check index before step 1
                    if not self.parameters.pure_exploration and all_path_simulation[individual_id][1][check_index] == 1: break #Check index led to observation of reward (reset exploration) If pure exploration is turned on, this is irrelevant
                    if all_path_simulation[individual_id][0][check_index] == single_path:
                        is_explore = False
                        break
                if is_explore: training_fitnesses[individual_id] += self.parameters.explore_reward

        return training_fitnesses, real_fitness

    def evolve(self, gen):
        tr_best_gen_fitness = [0]*parameters.num_agents

        # Reset all agent's fitness and sampling pool
        for agent_pop in self.all_agents:
            agent_pop.reset()
        team_ind = np.zeros(self.parameters.num_agents).astype(int)  # Team definitions by index

        #Generate training map and solution to test on for the epoch
        set_train_x, set_train_y = self.get_training_maze(self.parameters.num_train_evals)


        # MAIN LOOP
        for _ in range(self.parameters.population_size * self.parameters.num_evals_ccea):  # For evaluation

            # PICK TEAMS
            for i, agent_pop in enumerate(self.all_agents):
                choice = randint(0, len(agent_pop.selection_pool)-1)
                team_ind[i] = agent_pop.selection_pool.pop(choice)

            # SIMULATION AND TRACK REWARD
            fitnesses = [0]*self.parameters.num_agents
            for train_x, train_y in zip(set_train_x, set_train_y): #For all maps in the training set
                train_fitnesses, _ = self.compute_fitness(team_ind, train_x, train_y)
                fitnesses = map(add, fitnesses, train_fitnesses/self.parameters.num_train_evals )  # Returns rewards for each member of the team for each individual training map


            # ENCODE fitness back to each agent populations
            for i, agent_pop in enumerate(self.all_agents):
                agent_pop.fitness_evals[team_ind[i]] += fitnesses[i]/self.parameters.num_evals_ccea #Average not leniency

        #####Get champion index and run champion indiviual team
        champion_team = []
        #Find the champion idnex and bext performance
        for agent_id, agent_pop in enumerate(self.all_agents):
            tr_best_gen_fitness[agent_id] = max(agent_pop.fitness_evals) #Fix best performance
            agent_pop.best_index = agent_pop.fitness_evals.index(max(agent_pop.fitness_evals))
            champion_team.append(agent_pop.best_index)

        #Run simulation of champion team
        set_test_x, set_test_y = self.get_training_maze(self.parameters.num_train_evals*2)
        champ_train_fitnesses = [0] * self.parameters.num_agents
        champ_real_fitness = 0.0
        for test_x, test_y in zip(set_test_x, set_test_y):
            train_fitnesses, real_fitness = self.compute_fitness(champion_team, test_x, test_y)
            champ_real_fitness += real_fitness/len(set_test_x)
            champ_train_fitnesses = map(add, train_fitnesses/len(set_test_x), champ_train_fitnesses)  # Returns rewards for each member of the team for each individua


        #Save population and HOF
        if gen % 100 == 0:
            for pop_ind, agent_pop in enumerate(self.all_agents):
                if agent_pop.is_static: continue #Don't Save static populations

                ig_folder = self.parameters.save_foldername + '/Agent_' + str(pop_ind) + '/'
                if not os.path.exists(ig_folder): os.makedirs(ig_folder)

                for individial_ind, individual in enumerate(agent_pop.pop): #Save population
                    self.save(individual, ig_folder + str(individial_ind))

                self.save(agent_pop.pop[champion_team[pop_ind]], self.parameters.save_foldername + 'champion_' + str(pop_ind)) #Save champion
                np.savetxt(self.parameters.save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        #SSNE Epoch: Selection and Mutation/Crossover step
        for agent_id, agent_pop in enumerate(self.all_agents):
            if not agent_pop.is_static:
                self.ssne.epoch(agent_pop.pop, agent_pop.fitness_evals)

        #print self.all_agents[0].pop[champion_team[0]].fast_net.w_inp
        test = np.array(self.all_agents[0].fitness_evals)
        #print np.min(test), 'Max: ', np.max(test), 'Avg: ', np.mean(test), 'STD: ', np.std(test)
        return tr_best_gen_fitness, champ_train_fitnesses, champ_real_fitness

    def get_training_maze(self, num_examples):
        set_x = []; set_y = []
        for _ in range(num_examples):

            # Distraction/Hallway parts
            train_x = []; train_y = [[],[],[]]
            for junction in range(self.parameters.depth):
                corridor_len = randint(self.parameters.corridor_bound[0], self.parameters.corridor_bound[1])
                for i in range(corridor_len-1, -1, -1):
                    train_x.append(i)
                for div in train_y:
                    div.append(1 if random.random()<0.5 else -1)

            #Make sure the target (goal location) for each division aren't the same
            if train_y[0] == train_y[1]: train_y[0][randint(0, self.parameters.depth -1)] *= -1
            if train_y[1] == train_y[2]: train_y[2][randint(0, self.parameters.depth - 1)] *= -1

            set_x.append(train_x); set_y.append(train_y)

        return set_x, set_y



if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Multi-Agent TMaze Training ', parameters.arch_type

    sim_task = Task_Multi_TMaze(parameters)
    for gen in range(1, parameters.total_gens):
        gen_best_fitnesses, champ_train_fitnesses, champ_real_fitness = sim_task.evolve(gen)
        print 'Gen:', gen, ' #Trials:', parameters.num_trials, ' Epoch_best:', ['%.2f' % i for i in gen_best_fitnesses], ' Champ_train:', ['%.2f' % i for i in champ_train_fitnesses], ' Champ_real:', '%.2f'%champ_real_fitness#, ' Valid Score:', "%0.2f" % valid_score, '  Cumul_Valid_Score:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(sum(champ_train_fitnesses)/parameters.num_agents, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(champ_real_fitness, gen)  # Add best global performance to tracker














