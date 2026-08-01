from dependency import *  # noqa: F401,F403


def model_evaluation_7(custom_prediction, threshold):
    print("\nPrediction using Threshold =", threshold)

    print(custom_prediction[:20])

    # --------------------------------
    # Sigmoid Function
    # --------------------------------

    x = np.linspace(-10,10,200)

    sigmoid = 1/(1+np.exp(-x))

    plt.figure(figsize=(7,5))

    plt.plot(x,sigmoid)

    plt.title("Sigmoid Function")

    plt.xlabel("x")

    plt.ylabel("Sigmoid(x)")

    plt.grid(True)

    plt.show()
