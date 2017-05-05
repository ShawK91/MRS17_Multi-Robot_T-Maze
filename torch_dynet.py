import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import random
import numpy as np, sys


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

        #Adaptive state components
        self.hidden_act = Variable(torch.zeros(1, memory_size), requires_grad=0)
        self.mem = Variable(torch.zeros(1, memory_size), requires_grad=0)
        self.rec_output = Variable(torch.zeros(1, output_size), requires_grad=0)

        #Optimization variables ignore
        self.add_one = Variable(torch.ones(1, 1), requires_grad=0)


    def pad_one(self, input):
        return torch.cat((input,self.add_one), 1 )

    def kforward(self, input):
        #Pad 1 to consider for bias
        # input = self.pad_one(input)
        # self.mem = self.pad_one(self.mem)
        # self.rec_output = self.pad_one(self.rec_output)

        #Input gate
        ig1 = input.mm(self.w_inpgate)
        ig2 = self.mem.mm(self.w_mem_inpgate)
        ig3 = self.rec_output.mm(self.w_rec_inpgate)
        input_gate_out = torch.nn.functional.sigmoid(ig1 + ig2 + ig3)



        #Input Processing
        ig1 = input.mm(self.w_inp)
        ig2 = self.rec_output.mm(self.w_rec_inp)
        block_inp_out = torch.nn.functional.sigmoid(ig1 + ig2)

        #Input sum out after appluting input gate
        input_out = input_gate_out * block_inp_out

        #Read Gate
        ig1 = input.mm(self.w_readgate)
        ig2 = self.mem.mm(self.w_mem_readgate)
        ig3 = self.rec_output.mm(self.w_rec_readgate)
        read_gate_out = torch.nn.functional.sigmoid(ig1 + ig2 + ig3)
        memory_read = read_gate_out * self.mem

        #Compute hidden activation
        hidden_act = memory_read + input_out

        #Write (Update memory)
        ig1 = input.mm(self.w_writegate)
        ig2 = self.mem.mm(self.w_mem_writegate)
        ig3 = self.rec_output.mm(self.w_rec_writegate)
        write_gate_out = torch.nn.functional.sigmoid(ig1 + ig2 + ig3)
        self.mem = self.mem + write_gate_out

        #Compute final output
        output = hidden_act.mm(self.w_hid_out)

        #Update last output and return
        self.rec_output = output
        return output

    def reset_state(self):
        self.hidden_act = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=0)
        self.rec_output = Variable(torch.zeros(1, self.output_size), requires_grad=0)

    def kforward(self, input):
        # Pad 1 to consider for bias
        # input = self.pad_one(input)
        # self.mem = self.pad_one(self.mem)
        # self.rec_output = self.pad_one(self.rec_output)



        # Compute hidden activation
        self.hidden_act = torch.nn.functional.sigmoid(input.mm(self.w_readgate) + self.rec_output.mm(self.w_rec_readgate) + self.mem.mm(self.w_mem_readgate)) * self.mem + torch.nn.functional.sigmoid(input.mm(self.w_inpgate) + self.mem.mm(self.w_mem_inpgate) + self.rec_output.mm(self.w_rec_inpgate)) * torch.nn.functional.sigmoid(input.mm(self.w_inp) + self.rec_output.mm(self.w_rec_inp))

        self.mem = self.mem + torch.nn.functional.sigmoid(input.mm(self.w_writegate) + self.mem.mm(self.w_mem_writegate) + self.rec_output.mm(self.w_rec_writegate))

        # Compute final output
        output = self.hidden_act.mm(self.w_hid_out)
        #print self.mem
        #TODO change last output


        return output

    def forward(self, input):
        return nn.functional.sigmoid(input.mm(self.w_inp)).mm(self.w_hid_out)

if __name__ == '__main__':
    seq_len = 3
    num_train_examples = 1000
    num_epochs = 100


    train_x, train_y = test_sequence_data(num_examples=num_train_examples, seq_len=seq_len)




    model = GRUMB(1, 5, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(num_epochs):
        for example in range(len(train_x)):
            model.reset_state()
            for item in train_x[example]:
                x = Variable(torch.Tensor([item]), requires_grad=True)
                x = x.unsqueeze(0)
                out = model.forward(x)

            #sys.exit()
            #Compare with target and compute loss
            y = Variable(torch.Tensor(train_y[example]))
            y = y.unsqueeze(0)
            #print y


            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward(retain_variables = True)
            optimizer.step()
        #print 'Epoch: ', loss






    #TEST
    #test_x, test_y = test_sequence_data(num_examples=num_training_examples, seq_len=seq_len)
    test_x, test_y = train_x, train_y
    score = 0.0
    for example in range(len(test_x)):
        for item in test_x[example]:
            x = Variable(torch.Tensor([item]), requires_grad=True)
            x = x.unsqueeze(0)
            out = model.forward(x)

        #Compare with target and compute loss
        y = Variable(torch.Tensor(train_y[example]))
        y = y.unsqueeze(0)

        y_scalar = y.data.numpy()[0][0]
        out_scalar = out.data.numpy()[0][0]

        if abs(y_scalar - out_scalar) < 0.49:
            score+= 1

    print score/len(test_x)




