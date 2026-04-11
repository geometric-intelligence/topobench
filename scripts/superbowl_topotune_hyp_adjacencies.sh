
# NOTE: Rank 3 adjacency (semantic clusters) is excluded because the 24 large
# clusters produce a very dense adjacency matrix that causes OOM on H100 GPUs.
# Use incidence neighborhoods for rank 3 instead.

seeds=(1 3 5)

for seed in ${seeds[@]}; do
    # Rank 2 adjacency (threads)
    python -m topobench \
        dataset=hypergraph/superbowl_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=128 \
        model.feature_encoder.proj_dropout=0.25 \
        model.feature_encoder.selected_dimensions=\[0,1,2,3\] \
        model.backbone.neighborhoods=\[2-up_adjacency-0\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=Superbowl_Analysis \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[0\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[Superbowl_TopoTune_Adj]" &

    # Rank 1 adjacency (authors)
    python -m topobench \
        dataset=hypergraph/superbowl_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=128 \
        model.feature_encoder.proj_dropout=0.25 \
        model.feature_encoder.selected_dimensions=\[0,1,2,3\] \
        model.backbone.neighborhoods=\[1-up_adjacency-0\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=Superbowl_Analysis \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[1\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[Superbowl_TopoTune_Adj]" &

    # Ranks 1+2 combined
    python -m topobench \
        dataset=hypergraph/superbowl_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=128 \
        model.feature_encoder.proj_dropout=0.25 \
        model.feature_encoder.selected_dimensions=\[0,1,2,3\] \
        model.backbone.neighborhoods=\[2-up_adjacency-0,1-up_adjacency-0\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=Superbowl_Analysis \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[2\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[Superbowl_TopoTune_Adj]"

done

wait
echo "All TopoTune adjacency runs completed."
