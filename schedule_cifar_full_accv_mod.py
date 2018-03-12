from train import Train

if __name__ == '__main__':

    path = "exp_cifar_full_accv_mod"

    train = Train()

    experiments = [
        # CIFAR10

        {
            "loss": "loss_accv_mod",
            "hash_size": 24,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv_mod",
            "hash_size": 16,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv_mod",
            "hash_size": 32,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv_mod",
            "hash_size": 48,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv_mod",
            "hash_size": 4,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv_mod",
            "hash_size": 8,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 50,
            "number_of_epochs_per_decay": 10.0,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
    ]

    for e in experiments:
        train.run(path, e)
