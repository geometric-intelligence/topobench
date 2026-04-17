# SCCN
python -m topobench \
    dataset=graph/MUTAG \
    model=simplicial/topotune_onehasse,simplicial/topotune \
    model.feature_encoder.out_channels=128 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1 \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=NoReadOut \
    logger.wandb.project=TopoTune_repSCCNone \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    transforms.graph2simplicial_lifting.signed=True \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.dataloader_params.batch_size=64 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    optimizer.parameters.lr=0.001 \
    --multirun &


python -m topobench \
    dataset=graph/NCI1 \
    model=simplicial/topotune_onehasse,simplicial/topotune \
    model.feature_encoder.out_channels=64 \
    model.backbone.GNN.num_layers=1 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=3 \
    model.feature_encoder.proj_dropout=0.5 \
    model.readout.readout_name=PropagateSignalDown \
    transforms.graph2simplicial_lifting.signed=True \
    dataset.dataloader_params.batch_size=128 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    trainer.devices=\[0\] \
    logger.wandb.project=TopoTune_repSCCNone \
    optimizer.parameters.lr=0.001 \
    --multirun &


python -m topobench \
    dataset=graph/NCI109 \
    model=simplicial/topotune_onehasse,simplicial/topotune \
    model.feature_encoder.out_channels=64 \
    model.backbone.GNN.num_layers=1 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\] \
    model.backbone.layers=4 \
    model.readout.readout_name=NoReadOut \
    transforms.graph2simplicial_lifting.signed=True \
    model.feature_encoder.proj_dropout=0.25 \
    dataset.dataloader_params.batch_size=128 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    trainer.devices=\[1\] \
    logger.wandb.project=TopoTune_repSCCNone \
    optimizer.parameters.lr=0.001 \
    --multirun &



python -m topobench \
    model=simplicial/topotune_onehasse,simplicial/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\] \
    dataset=graph/PROTEINS \
    optimizer.parameters.lr=0.01 \
    model.feature_encoder.out_channels=128 \
    model.backbone.layers=3 \
    model.readout.readout_name=NoReadOut \
    transforms.graph2simplicial_lifting.signed=True \
    model.feature_encoder.proj_dropout=0.5 \
    dataset.dataloader_params.batch_size=128 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    trainer.devices=\[1\] \
    logger.wandb.project=TopoTune_repSCCNone \
    --multirun &


python -m topobench \
    model=simplicial/topotune_onehasse,simplicial/topotune \
    dataset=graph/ZINC \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\] \
    optimizer.parameters.lr=0.001 \
    model.feature_encoder.out_channels=128 \
    model.backbone.layers=4 \
    model.readout.readout_name=PropagateSignalDown \
    transforms.graph2simplicial_lifting.signed=True \
    model.feature_encoder.proj_dropout=0.5 \
    dataset.dataloader_params.batch_size=128 \
    callbacks.early_stopping.min_delta=0.005 \
    transforms.one_hot_node_degree_features.degrees_fields=x \
    seed=42,3,5,23,150 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=TopoTune_repSCCNone \
    trainer.devices=\[0\] \
    --multirun &

python -m topobench \
    model=simplicial/topotune_onehasse,simplicial/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\] \
    dataset=graph/cocitation_citeseer \
    optimizer.parameters.lr=0.01 \
    model.feature_encoder.out_channels=64 \
    model.backbone.layers=2 \
    model.readout.readout_name=NoReadOut \
    transforms.graph2simplicial_lifting.signed=True \
    model.feature_encoder.proj_dropout=0.5 \
    dataset.dataloader_params.batch_size=1 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    logger.wandb.project=TopoTune_repSCCNone \
    trainer.devices=\[0\] \
    --multirun &

python -m topobench \
    model=simplicial/topotune_onehasse,simplicial/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN._target_=topobench.nn.backbones.graph.IdentityGCN \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\] \
    dataset=graph/cocitation_cora \
    optimizer.parameters.lr=0.01 \
    model.feature_encoder.out_channels=32 \
    model.backbone.layers=2 \
    model.readout.readout_name=NoReadOut \
    transforms.graph2simplicial_lifting.signed=True \
    model.feature_encoder.proj_dropout=0.5 \
    dataset.dataloader_params.batch_size=1 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    logger.wandb.project=TopoTune_repSCCNone \
    trainer.devices=\[1\] \
    --multirun &

python -m topobench \
    model=simplicial/topotune_onehasse,simplicial/topotune \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-up_incidence-0,1-down_incidence-1,1-down_laplacian-1,1-up_laplacian-1,1-up_incidence-1,1-down_incidence-2,1-down_laplacian-2\] \
    dataset=graph/cocitation_pubmed \
    optimizer.parameters.lr=0.01 \
    model.feature_encoder.out_channels=64 \
    model.backbone.layers=2 \
    model.readout.readout_name=NoReadOut \
    transforms.graph2simplicial_lifting.signed=True \
    model.feature_encoder.proj_dropout=0.5 \
    dataset.dataloader_params.batch_size=1 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    logger.wandb.project=TopoTune_repSCCNone \
    trainer.devices=\[1\] \
    --multirun
