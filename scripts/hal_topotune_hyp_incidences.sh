
seeds=(1 3 5)

for seed in ${seeds[@]}; do
    # Rank 3 incidence (semantic clusters -> documents)
    python -m topobench \
        dataset=hypergraph/hal_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=128 \
        model.feature_encoder.proj_dropout=0.25 \
        model.feature_encoder.selected_dimensions=\[0,1,2,3\] \
        model.backbone.neighborhoods=\[3-down_incidence-3\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\],\[3\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=HAL_Analysis \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[0\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[HAL_TopoTune_Inc]" \
        --multirun &

    # Rank 2 incidence (institutions -> documents)
    python -m topobench \
        dataset=hypergraph/hal_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=128 \
        model.feature_encoder.proj_dropout=0.25 \
        model.feature_encoder.selected_dimensions=\[0,1,2,3\] \
        model.backbone.neighborhoods=\[2-down_incidence-2\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\],\[2\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=HAL_Analysis \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[1\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[HAL_TopoTune_Inc]" \
        --multirun &

    # Rank 1 incidence (authors -> documents)
    python -m topobench \
        dataset=hypergraph/hal_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=128 \
        model.feature_encoder.proj_dropout=0.25 \
        model.feature_encoder.selected_dimensions=\[0,1,2,3\] \
        model.backbone.neighborhoods=\[1-down_incidence-1\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\],\[1\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=HAL_Analysis \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[2\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[HAL_TopoTune_Inc]" \
        --multirun &

    # All ranks combined
    python -m topobench \
        dataset=hypergraph/hal_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=128 \
        model.feature_encoder.proj_dropout=0.25 \
        model.feature_encoder.selected_dimensions=\[0,1,2,3\] \
        model.backbone.neighborhoods=\[3-down_incidence-3,2-down_incidence-2,1-down_incidence-1\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\],\[3,2,1\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=HAL_Analysis \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[3\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[HAL_TopoTune_Inc]" \
        --multirun

done

wait
echo "All TopoTune incidence runs completed."
