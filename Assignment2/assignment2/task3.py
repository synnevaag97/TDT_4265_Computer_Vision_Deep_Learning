import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer, calculate_accuracy


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)


    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!
    # Weight
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = False

    model_w = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_w = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_w, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_w, val_history_w = trainer_w.train(num_epochs)

    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model_w))

    #Sigmoid
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False

    model_s = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_s = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_s, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_s, val_history_s = trainer_s.train(num_epochs)
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model_s))

    # Momentum
    learning_rate = .02

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    model_m = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_m = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_m, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_m, val_history_m = trainer_m.train(num_epochs)

    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model_m))

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_w["loss"], "Task 2 Model with improved weights", npoints_to_average=10)
    utils.plot_loss(
        train_history_s["loss"], "Task 2 Model with improved weights and sigmoid", npoints_to_average=10)
    utils.plot_loss(
        train_history_m["loss"], "Task 2 Model with improved weights and sigmoid and momentum", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.895, .965])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(val_history_w["accuracy"], "Task 2 Model with improved weights")
    utils.plot_loss(val_history_s["accuracy"], "Task 2 Model with improved weights and sigmoid")
    utils.plot_loss(val_history_m["accuracy"], "Task 2 Model with improved weights and sigmoid and momentum")
    plt.ylabel("Validation Accuracy")
    plt.legend(loc='lower right')
    plt.savefig("task3_comparison.png")
    plt.show()
