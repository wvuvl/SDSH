from train import Train

if __name__ == '__main__':

    path = "exp_cifar_full_marginloss"

    train = Train()

    experiments = [
        # CIFAR10
        {
            "loss": "marginloss",
            "hash_size": 24,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "marginloss",
            "hash_size": 24,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "marginloss",
            "hash_size": 16,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "marginloss",
            "hash_size": 32,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "marginloss",
            "hash_size": 48,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "marginloss",
            "hash_size": 8,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "marginloss",
            "hash_size": 12,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
    ]

    for e in experiments:
        train.run(path, e)
