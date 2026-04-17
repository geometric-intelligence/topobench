python -m topobench \
    model=cell/topotune_onehasse,cell/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1 \
    model.backbone.neighborhoods=\[1-up_incidence-0,1-up_adjacency-1,1-down_incidence-2\] \
    logger.wandb.project=TopoTune_CWN \
    dataset=graph/MUTAG \
    optimizer.parameters.lr=0.001 \
    model.feature_encoder.out_channels=128 \
    model.backbone.layers=2 \
    model.readout.readout_name=PropagateSignalDown \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.dataloader_params.batch_size=32 \
    transforms.graph2cell_lifting.max_cell_length=10 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    trainer.devices=\[1\] \
    --multirun &

python -m topobench \
    model=cell/topotune_onehasse,cell/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1 \
    model.backbone.neighborhoods=\[1-up_incidence-0,1-up_adjacency-1,1-down_incidence-2\] \
    logger.wandb.project=TopoTune_CWN \
    dataset=graph/NCI1 \
    optimizer.parameters.lr=0.001 \
    model.feature_encoder.out_channels=64 \
    model.backbone.layers=4 \
    model.readout.readout_name=PropagateSignalDown \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.dataloader_params.batch_size=128 \
    transforms.graph2cell_lifting.max_cell_length=10 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.devices=\[1\] \
    --multirun &


python -m topobench \
    model=cell/topotune_onehasse,cell/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1 \
    model.backbone.neighborhoods=\[1-up_incidence-0,1-up_adjacency-1,1-down_incidence-2\] \
    logger.wandb.project=TopoTune_CWN \
    dataset=graph/NCI109 \
    optimizer.parameters.lr=0.001 \
    model.feature_encoder.out_channels=128 \
    model.backbone.layers=3 \
    model.readout.readout_name=PropagateSignalDown \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.dataloader_params.batch_size=128 \
    transforms.graph2cell_lifting.max_cell_length=10 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.devices=\[2\] \
    --multirun &

python -m topobench \
    model=cell/topotune_onehasse,cell/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1 \
    model.backbone.neighborhoods=\[1-up_incidence-0,1-up_adjacency-1,1-down_incidence-2\] \
    logger.wandb.project=TopoTune_CWN \
    dataset=graph/ZINC \
    optimizer.parameters.lr=0.001 \
    model.feature_encoder.out_channels=64 \
    model.backbone.layers=2 \
    model.readout.readout_name=PropagateSignalDown \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.dataloader_params.batch_size=128 \
    transforms.graph2cell_lifting.max_cell_length=10 \
    callbacks.early_stopping.min_delta=0.005 \
    transforms.one_hot_node_degree_features.degrees_fields=x \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    trainer.devices=\[2\] \
    --multirun &


python -m topobench \
    model=cell/topotune_onehasse,cell/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1 \
    model.backbone.neighborhoods=\[1-up_incidence-0,1-up_adjacency-1,1-down_incidence-2\] \
    logger.wandb.project=TopoTune_CWN \
    dataset=graph/cocitation_citeseer \
    optimizer.parameters.lr=0.001 \
    model.feature_encoder.out_channels=128 \
    model.backbone.layers=2 \
    model.readout.readout_name=PropagateSignalDown \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.dataloader_params.batch_size=1 \
    transforms.graph2cell_lifting.max_cell_length=10 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[3\] \
    --multirun &

python -m topobench \
    model=cell/topotune_onehasse,cell/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1 \
    model.backbone.neighborhoods=\[1-up_incidence-0,1-up_adjacency-1,1-down_incidence-2\] \
    logger.wandb.project=TopoTune_CWN \
    dataset=graph/cocitation_pubmed \
    optimizer.parameters.lr=0.01 \
    model.feature_encoder.out_channels=64 \
    model.backbone.layers=1 \
    model.readout.readout_name=PropagateSignalDown \
    model.feature_encoder.proj_dropout=0.5 \
    dataset.dataloader_params.batch_size=1 \
    transforms.graph2cell_lifting.max_cell_length=10 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    trainer.devices=\[3\] \
    --multirun &


python -m topobench \
    model=cell/topotune_onehasse,cell/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1 \
    model.backbone.neighborhoods=\[1-up_incidence-0,1-up_adjacency-1,1-down_incidence-2\] \
    logger.wandb.project=TopoTune_CWN \
    dataset=graph/PROTEINS,graph/cocitation_cora \
    optimizer.parameters.lr=0.001 \
    model.feature_encoder.out_channels=128 \
    model.backbone.layers=4 \
    model.readout.readout_name=PropagateSignalDown \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.dataloader_params.batch_size=1 \
    trainer.max_epochs=1000 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    trainer.devices=\[4\] \
    --multirun
