from dependency import *  # noqa: F401,F403


def data_preparation_2():
    car_prices_array = [3,4,5,6,7,8,9]
    car_price_np = np.array(car_prices_array,dtype=np.float32)
    car_price_np = car_price_np.reshape(-1,1)
    car_price_tensor = Variable(torch.from_numpy(car_price_np))

    number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
    number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
    number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
    number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))

    return car_price_tensor, number_of_car_sell_tensor
