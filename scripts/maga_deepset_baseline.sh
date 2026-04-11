
python -m topobench \
    dataset=hypergraph/maga_arlequin \
    model=pointcloud/deepset \
    model.feature_encoder.out_channels=128 \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.split_params.data_seed=1 \
    dataset.loader.parameters.ho_init_method=avg_post \
    dataset.loader.parameters.max_posts_per_user=1000 \
    model.readout.readout_name=NoReadOut \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    logger.wandb.project=MAGA_Analysis_small \
    trainer.max_epochs=1000 \
    trainer.min_epochs=250 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=100 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=hypergraph/maga_arlequin \
    model=pointcloud/deepset \
    model.feature_encoder.out_channels=128 \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.split_params.data_seed=1 \
    dataset.loader.parameters.ho_init_method=avg_post \
    dataset.loader.parameters.max_posts_per_user=2000 \
    model.readout.readout_name=NoReadOut \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    logger.wandb.project=MAGA_Analysis_small \
    trainer.max_epochs=1000 \
    trainer.min_epochs=250 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=100 \
    tags="[FirstExperiments]" \
    --multirun &


python -m topobench \
    dataset=hypergraph/maga_arlequin \
    model=pointcloud/deepset \
    model.feature_encoder.out_channels=128 \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.split_params.data_seed=3 \
    dataset.loader.parameters.ho_init_method=avg_post \
    dataset.loader.parameters.max_posts_per_user=1000 \
    model.readout.readout_name=NoReadOut \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    logger.wandb.project=MAGA_Analysis_small \
    trainer.max_epochs=1000 \
    trainer.min_epochs=250 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=100 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=hypergraph/maga_arlequin \
    model=pointcloud/deepset \
    model.feature_encoder.out_channels=128 \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.split_params.data_seed=3 \
    dataset.loader.parameters.ho_init_method=avg_post \
    dataset.loader.parameters.max_posts_per_user=2000 \
    model.readout.readout_name=NoReadOut \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    logger.wandb.project=MAGA_Analysis_small \
    trainer.max_epochs=1000 \
    trainer.min_epochs=250 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=100 \
    tags="[FirstExperiments]" \
    --multirun &


python -m topobench \
    dataset=hypergraph/maga_arlequin \
    model=pointcloud/deepset \
    model.feature_encoder.out_channels=128 \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.split_params.data_seed=5 \
    dataset.loader.parameters.ho_init_method=avg_post \
    dataset.loader.parameters.max_posts_per_user=1000 \
    model.readout.readout_name=NoReadOut \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    logger.wandb.project=MAGA_Analysis_small \
    trainer.max_epochs=1000 \
    trainer.min_epochs=250 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=100 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=hypergraph/maga_arlequin \
    model=pointcloud/deepset \
    model.feature_encoder.out_channels=128 \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.split_params.data_seed=5 \
    dataset.loader.parameters.ho_init_method=avg_post \
    dataset.loader.parameters.max_posts_per_user=2000 \
    model.readout.readout_name=NoReadOut \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    logger.wandb.project=MAGA_Analysis_small \
    trainer.max_epochs=1000 \
    trainer.min_epochs=250 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=100 \
    tags="[FirstExperiments]" \
    --multirun &