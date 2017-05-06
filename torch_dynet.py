import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import random
import numpy as np, sys

#TODO Weight Initialization
#TODO Save/Load
#TODO Dynamic length seq_prediction


def test_sequence_data(num_examples, seq_len):
    train_x = []; train_y = []
    for example in range(num_examples):
        x = [];
        for pos in range(seq_len):
            if random.random()<0.5: x.append(1)
            else: x.append(-1)

        if sum(x) < 0: train_y.append([0])
        else: train_y.append([1])
        train_x.append(x)
    return train_x, train_y

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

        # #Adaptive state components
        # self.hidden_act = Variable(torch.zeros(1, memory_size), requires_grad=0)
        # self.mem = Variable(torch.zeros(1, memory_size), requires_grad=0)
        # self.rec_output = Variable(torch.zeros(1, output_size), requires_grad=0)
        #
        # #Optimization variables ignore
        # self.add_one = Variable(torch.ones(1, 1), requires_grad=0)


    def graph_compute(self, input, mem, rec_output):
        # Compute hidden activation
        hidden_act = torch.nn.functional.sigmoid(input.mm(self.w_readgate) + rec_output.mm(self.w_rec_readgate) + mem.mm(self.w_mem_readgate)) * mem + torch.nn.functional.sigmoid(input.mm(self.w_inpgate) + mem.mm(self.w_mem_inpgate) + rec_output.mm(self.w_rec_inpgate)) * torch.nn.functional.sigmoid(input.mm(self.w_inp) + rec_output.mm(self.w_rec_inp))

        #Update mem
        mem = mem + torch.nn.functional.sigmoid(input.mm(self.w_writegate) + mem.mm(self.w_mem_writegate) + rec_output.mm(self.w_rec_writegate))


        # Compute final output
        output = hidden_act.mm(self.w_hid_out)

        return output, mem


    def forward(self, input):
        mem = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        out = Variable(torch.zeros(1, self.output_size), requires_grad=0)

        for item in input:
            x = Variable(torch.Tensor([item]), requires_grad=True); x = x.unsqueeze(0)
            out, mem = model.graph_compute(x, mem, out)

        return out



if __name__ == '__main__':
    seq_len_train = 21
    seq_len_test = 21
    num_train_examples = 10000
    num_test_examples = 1000
    num_epochs = 100

    train_x, train_y = test_sequence_data(num_examples=num_train_examples, seq_len=seq_len_train)


    model = GRUMB(1, 5, 1)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.0
        for example in range(len(train_x)): #For all examples
            out = model.forward(train_x[example])

            #Compare with target and compute loss
            y = Variable(torch.Tensor(train_y[example])); y = y.unsqueeze(0)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward(retain_variables = True)
            optimizer.step()
            epoch_loss += loss.data.numpy()[0]
        print 'Epoch: ', epoch, ' Loss: ',  epoch_loss/len(train_x)



    #TEST
    test_x, test_y = test_sequence_data(num_examples=num_test_examples, seq_len=seq_len_test)
    #test_x, test_y = train_x, train_y
    score = 0.0
    for example in range(len(test_x)):
        out = model.forward(test_x[example])

        #Compare with target and compute loss
        y = Variable(torch.Tensor(test_y[example])); y = y.unsqueeze(0)
        y_scalar = y.data.numpy()[0][0]
        out_scalar = out.data.numpy()[0][0]

        if abs(y_scalar - out_scalar) < 0.45:
            score += 1
        #print y_scalar, out_scalar

    print 'Test Score: ', score/len(test_x)




