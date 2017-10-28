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
            "total_epoch_count": 100,
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
            "total_epoch_count": 100,
            "number_of_epochs_per_decay": 20.0,
            "weight_decay_factor": 0.0012,
            "learning_rate": 0.001,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv_mod",
            "hash_size": 24,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 200,
            "number_of_epochs_per_decay": 67.0,
            "weight_decay_factor": 5.0e-5,
            "learning_rate": 0.02,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_triplet",
            "hash_size": 24,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 200,
            "number_of_epochs_per_decay": 67.0,
            "weight_decay_factor": 5.0e-5,
            "learning_rate": 0.025,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_spring",
            "hash_size": 24,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 100,
            "number_of_epochs_per_decay": 67.0,
            "weight_decay_factor": 5.0e-5,
            "learning_rate": 0.020,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv",
            "hash_size": 12,
            "margin": 6,
            "batch_size": 150,
            "total_epoch_count": 100,
            "number_of_epochs_per_decay": 20.0,
            "weight_decay_factor": 0.0012,
            "learning_rate": 0.001,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv_mod",
            "hash_size": 12,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 200,
            "number_of_epochs_per_decay": 67.0,
            "weight_decay_factor": 5.0e-5,
            "learning_rate": 0.02,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_triplet",
            "hash_size": 12,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 200,
            "number_of_epochs_per_decay": 67.0,
            "weight_decay_factor": 5.0e-5,
            "learning_rate": 0.025,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_spring",
            "hash_size": 12,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 100,
            "number_of_epochs_per_decay": 67.0,
            "weight_decay_factor": 5.0e-5,
            "learning_rate": 0.020,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv",
            "hash_size": 32,
            "margin": 16,
            "batch_size": 150,
            "total_epoch_count": 100,
            "number_of_epochs_per_decay": 20.0,
            "weight_decay_factor": 0.0012,
            "learning_rate": 0.001,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_accv_mod",
            "hash_size": 32,
            "margin": 0.5,
            "batch_size": 150,
            "total_epoch_count": 200,
            "number_of_epochs_per_decay": 67.0,
            "weight_decay_factor": 5.0e-5,
            "learning_rate": 0.02,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_triplet",
            "hash_size": 32,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 200,
            "number_of_epochs_per_decay": 67.0,
            "weight_decay_factor": 5.0e-5,
            "learning_rate": 0.025,
            "learning_rate_decay_factor": 2.0 / 3.0
        },
        {
            "loss": "loss_spring",
            "hash_size": 32,
            "margin": 2.0,
            "batch_size": 150,
            "total_epoch_count": 100,
            "number_of_epochs_per_decay": 67.0,
            "weight_decay_factor": 5.0e-5,
            "learning_rate": 0.020,
            "learning_rate_decay_factor": 2.0 / 3.0
        }
    ]

    for e in experiments:
        train.run(path, e)
