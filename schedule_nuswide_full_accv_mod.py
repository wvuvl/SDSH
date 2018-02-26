from train import Train

if __name__ == '__main__':

    path = "exp_nuswide_full_accv_mod"

    train = Train()

    experiments = [
        # NUS_WIDE 2100._
        {
            "dataset": "nus2100._",
            "loss": "loss_accv_mod",
            "hash_size": 24,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 5e-05,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "dataset": "nus2100._",
            "loss": "loss_accv_mod",
            "hash_size": 16,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 5e-05,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "dataset": "nus2100._",
            "loss": "loss_accv_mod",
            "hash_size": 32,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 5e-05,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "dataset": "nus2100._",
            "loss": "loss_accv_mod",
            "hash_size": 48,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 28,
            "number_of_epochs_per_decay": 30,
            "weight_decay_factor": 5e-05,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
    ]

    for e in experiments:
        train.run(path, e)
