from dependency import *  # noqa: F401,F403


def model_generation_3(car_price_tensor, iteration_number, loss_list, model, mse, number_of_car_sell_tensor, optimizer):
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

