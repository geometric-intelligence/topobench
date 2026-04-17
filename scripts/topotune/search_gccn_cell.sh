python -m topobench \
    dataset=graph/NCI109 \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-down_laplacian-1,1-up_laplacian-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-1,1-up_laplacian-1,1-down_incidence-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\],\[1-up_laplacian-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/cocitation_cora \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-down_laplacian-1,1-up_laplacian-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-1,1-up_laplacian-1,1-down_incidence-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\],\[1-up_laplacian-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/PROTEINS \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-down_laplacian-1,1-up_laplacian-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-1,1-up_laplacian-1,1-down_incidence-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\],\[1-up_laplacian-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/MUTAG \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-down_laplacian-1,1-up_laplacian-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-1,1-up_laplacian-1,1-down_incidence-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\],\[1-up_laplacian-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/ZINC \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-down_laplacian-1,1-up_laplacian-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-1,1-up_laplacian-1,1-down_incidence-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\],\[1-up_laplacian-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    transforms.one_hot_node_degree_features.degrees_fields=x \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/cocitation_citeseer \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-down_laplacian-1,1-up_laplacian-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-1,1-up_laplacian-1,1-down_incidence-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\],\[1-up_laplacian-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[4\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/NCI1 \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-down_laplacian-1,1-up_laplacian-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-1,1-up_laplacian-1,1-down_incidence-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\],\[1-up_laplacian-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/cocitation_pubmed \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-down_laplacian-1,1-up_laplacian-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-1,1-up_laplacian-1,1-down_incidence-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\],\[1-up_laplacian-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[7\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/NCI109 \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1,1-down_laplacian-1\],\[1-up_laplacian-0,1-down_incidence-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/cocitation_cora \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1,1-down_laplacian-1\],\[1-up_laplacian-0,1-down_incidence-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/PROTEINS \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1,1-down_laplacian-1\],\[1-up_laplacian-0,1-down_incidence-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/MUTAG \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1,1-down_laplacian-1\],\[1-up_laplacian-0,1-down_incidence-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/cocitation_citeseer \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1,1-down_laplacian-1\],\[1-up_laplacian-0,1-down_incidence-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[4\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/NCI1 \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1,1-down_laplacian-1\],\[1-up_laplacian-0,1-down_incidence-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobench \
    dataset=graph/cocitation_pubmed \
    model=cell/topotune,cell/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-up_laplacian-1,1-down_laplacian-1\],\[1-up_laplacian-0,1-down_incidence-2\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_cell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[7\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun
