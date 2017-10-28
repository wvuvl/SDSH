from train import Train

if __name__ == '__main__':

    path = "experiments"

    train = Train()

    experiments = [
        {
            "loss": "loss_accv",
            "hash_size": 24,
            "margin": 12,
            "batch_size": 150,
            "total_epoch_count": 4,
            "number_of_epochs_per_decay": 20.0,
            "weight_decay_factor": 0.0011,
            "learning_rate": 0.000225,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv",
            "hash_size": 24,
            "margin": 12,
            "batch_size": 150,
            "total_epoch_count": 4,
            "number_of_epochs_per_decay": 20.0,
            "weight_decay_factor": 0.0012,
            "learning_rate": 0.001,
            "learning_rate_decay_factor": 2.0 / 3.0
        }
    ]

    for e in experiments:
        train.run(path, e)
