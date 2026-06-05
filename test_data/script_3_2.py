import numpy as np
import matplotlib.pyplot as plt
import torch      
from torch.autograd import Variable     
import torch.nn as nn 
import warnings
warnings.filterwarnings("ignore")

input_dim = 1
output_dim = 1

# create class
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(LinearRegression,self).__init__()
        # Linear function.
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x)
    
def run_linear_regression_example():
    # define model
    model = LinearRegression(input_dim,output_dim) # input and output size are 1

    # MSE
    mse = nn.MSELoss()

    # Optimization (find parameters that minimize error)
    learning_rate = 0.02   # how fast we reach best parameters
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

    # train model
    loss_list = []
    iteration_number = 1001

    car_prices_array = [3,4,5,6,7,8,9]
    car_price_np = np.array(car_prices_array,dtype=np.float32)
    car_price_np = car_price_np.reshape(-1,1)
    car_price_tensor = Variable(torch.from_numpy(car_price_np))

    number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
    number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
    number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
    number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))

    for iteration in range(iteration_number):
            
        # optimization
        optimizer.zero_grad() 
        
        # Forward to get output
        results = model(car_price_tensor)
        
        # Calculate Loss
        loss = mse(results, number_of_car_sell_tensor)
        
        # backward propagation
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        # store loss
        loss_list.append(loss.data)
        
        # print loss
        if(iteration % 50 == 0):
            print('epoch {}, loss {}'.format(iteration, loss.data))

    plt.plot(range(iteration_number),loss_list)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.show()