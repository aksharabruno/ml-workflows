from dependency import *  # noqa: F401,F403


def model_evaluation_4(iteration_number, loss_list):
    plt.plot(range(iteration_number),loss_list)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.show()
