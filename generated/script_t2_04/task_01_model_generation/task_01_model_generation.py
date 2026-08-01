from dependency import *  # noqa: F401,F403


def model_generation_1():
    # create class
    class LinearRegression(nn.Module):
        def __init__(self,input_size,output_size):
            # super function. It inherits from nn.Module and we can access everythink in nn.Module
            super(LinearRegression,self).__init__()
            # Linear function.
            self.linear = nn.Linear(input_dim,output_dim)

        def forward(self,x):
            return self.linear(x)

    model = LinearRegression(input_dim,output_dim) # input and output size are 1

    # MSE
    mse = nn.MSELoss()

    # Optimization (find parameters that minimize error)
    learning_rate = 0.02   # how fast we reach best parameters
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

    # train model
    loss_list = []
    iteration_number = 1001

    return iteration_number, loss_list, model, mse, optimizer
