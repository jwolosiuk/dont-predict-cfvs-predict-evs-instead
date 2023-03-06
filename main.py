from training import TrainSchedule, TrainConfig

batch_size = 2
learning_rate = 3e-4
n_epochs = 400
dropout = 0

for dataset_id, n_buckets, n_neurons_in_layer in [(1, 1000, 500), (1, 1000, 1024), (1, 2000, 1024), (2, 1000, 1024), (2, 2000, 1024)]:
    for modification in [True, False]:
        for n_layers in [7, 5, 3, 2]:
            config = TrainConfig(hidden_sizes=[n_neurons_in_layer]*n_layers,
                                 is_ev_based=modification,
                                 dropout=dropout,
                                 batch_size=batch_size,
                                 lr=learning_rate,
                                 epochs=n_epochs,
                                 n_buckets=n_buckets,
                                 dataset_id=dataset_id)
            trainer = TrainSchedule(config)
            trainer.fit()
