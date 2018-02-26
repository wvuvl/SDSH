from train import Train

if __name__ == '__main__':

    path = "exp_nuswide_full_marginloss"

    train = Train()

    experiments = [
        # NUS_WIDE 2100._
        {
            "dataset": "nus2100._",
            "loss": "marginloss",
            "hash_size": 24,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "dataset": "nus2100._",
            "loss": "marginloss",
            "hash_size": 16,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "dataset": "nus2100._",
            "loss": "marginloss",
            "hash_size": 32,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "dataset": "nus2100._",
            "loss": "marginloss",
            "hash_size": 12,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "dataset": "nus2100._",
            "loss": "marginloss",
            "hash_size": 8,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "dataset": "nus2100._",
            "loss": "marginloss",
            "hash_size": 48,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 0.001,
            "learning_rate": 0.07 / 4.0,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
    ]

    for e in experiments:
        train.run(path, e)
