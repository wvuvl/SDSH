from train import Train

if __name__ == '__main__':

    path = "exp_imagenet_accv_mod"

    train = Train()

    experiments = [
        # Imagenet
        {
            "dataset": "imagenet",
            "loss": "loss_accv_mod",
            "hash_size": 24,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 130,
            "number_of_epochs_per_decay": 100,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0,
            "freeze": True,
        },
        {
            "dataset": "imagenet",
            "loss": "loss_accv_mod",
            "hash_size": 16,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 130,
            "number_of_epochs_per_decay": 100,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0,
            "freeze": True,
        },
        {
            "dataset": "imagenet",
            "loss": "loss_accv_mod",
            "hash_size": 32,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 130,
            "number_of_epochs_per_decay": 100,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0,
            "freeze": True,
        },
        {
            "dataset": "imagenet",
            "loss": "loss_accv_mod",
            "hash_size": 8,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 130,
            "number_of_epochs_per_decay": 100,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0,
            "freeze": True,
        },
    ]

    for e in experiments:
        train.run(path, e)
