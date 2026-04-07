
seeds=(1 3 5)

for seed in ${seeds[@]}; do
    python -m topobench \
        dataset=hypergraph/hal_arlequin \
        model=pointcloud/deepset \
        model.feature_encoder.out_channels=128 \
        model.feature_encoder.proj_dropout=0.25 \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=NoReadOut \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=HAL_Analysis \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[0\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[HAL_Baseline]" \
        --multirun &
done

wait
echo "All DeepSet baseline runs completed."
