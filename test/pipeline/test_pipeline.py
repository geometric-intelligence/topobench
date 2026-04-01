"""Test pipeline for a particular dataset and model."""

import hydra
import lightning as pl
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.loss.loss import TBLoss
from topobench.model.model import TBModel
from topobench.evaluator.evaluator import TBEvaluator
from topobench.nn.readouts import identical
from topobench.optimizer import TBOptimizer


class TestPipeline:
    """Test pipeline for a particular dataset and model."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_pipeline(self):
        """Test pipeline."""

        # configs
        config_dataset = OmegaConf.load("configs/dataset/hypergraph/chordonomicon.yaml")
        config_dataset.split_params.data_split_dir = f"datasets/data_splits/chordonomicon/{config_dataset.loader.parameters.version}"  # pylint: disable=line-too-long
        config_dataset.loader.parameters.data_dir = "datasets/hypergraph/chords"
        config_evaluator = {"task": "regression",
                            "num_classes": config_dataset.parameters.num_classes,
                            "metrics": ["rmse", "mse", "mae"]}
        config_loss = {"dataset_loss":
            {
                "task": "regression", 
                "loss_type": "mse"
                }
            }
        config_readout = {
            "hidden_dim": config_dataset.parameters.num_classes,
            "out_channels": config_dataset.parameters.num_classes,
            "task_level": config_dataset.parameters.task_level,
            "logits_linear_layer": False,
            }
        config_optimizer = {"optimizer_id": "Adam",
                            "parameters":
                                {"lr": 0.01,"weight_decay": 0.0005}
                            }

        # backbone class definition
        class ModelPipeLine(pl.LightningModule):
            """Custom model pipeline for testing.

            Parameters
            ----------
            dim_in_node : int
                Dimension of input node features.
            dim_hidden : int
                Dimension of hidden layers.
            dim_out : int
                Dimension of output features.
            """
            def __init__(self,
                        dim_in_node,  #batch.x.size(0)+batch.x_hyperedges.shape[1]
                        dim_hidden,
                        dim_out,
                        ):
                super().__init__()
                self.dim_hidden = dim_hidden
                self.linear_node_0 = torch.nn.Linear(dim_in_node, dim_hidden)
                self.linear_hyperedge_0 = torch.nn.Linear(dim_hidden, dim_out)

            def forward(self, batch):  #pylint: disable=arguments-differ
                """Forward pass.

                Parameters
                ----------
                batch : torch_geometric.data.Data
                    Input batch of data.

                Returns
                -------
                dict
                    Output dictionary containing node representation and hyperedge logits.
                """
                x_node = torch.concat((batch.x,
                                torch.sparse.mm(batch.incidence_hyperedges, batch.x_hyperedges)),  #pylint: disable=not-callable
                                      dim=1)
                h_node = self.linear_node_0(x_node)
                h_node = torch.relu(h_node)
                h_hyperedge = torch.mm(batch.incidence_hyperedges.T, h_node)
                h_hyperedge = self.linear_hyperedge_0(h_hyperedge)
                model_out =  {'h_node': h_node,
                            'h_hyperedge': h_hyperedge,
                            "labels": batch.y_hyperedges}
                model_out["logits"] = model_out["h_hyperedge"]
                return model_out

        # dataset
        dataset_loader = instantiate(config_dataset.loader)
        dataset, dataset_dir = dataset_loader.load()
        preprocessor = PreProcessor(dataset, dataset_dir)
        dataset_train, dataset_val, dataset_test = preprocessor.load_dataset_splits(config_dataset.split_params)  #pylint: disable=line-too-long
        datamodule = TBDataloader(
                    dataset_train=dataset_train,
                    dataset_val=dataset_val,
                    dataset_test=dataset_test,
                    **config_dataset.get("dataloader_params", {}),
                )

        # model
        input_dim = config_dataset.parameters.num_edge_features
        if config_dataset.loader.parameters.version == "single_scale":
            input_dim += config_dataset.parameters.num_node_features_single_scale
        elif config_dataset.loader.parameters.version == "all_scales":
            input_dim += config_dataset.parameters.num_node_features_all_scales
        backbone = ModelPipeLine(dim_in_node=input_dim,
                                dim_hidden=10,
                                dim_out=config_dataset.parameters.num_classes)
        loss = TBLoss(config_loss["dataset_loss"])
        optimizer = TBOptimizer(**config_optimizer)
        readout = identical.NoReadOut(**config_readout)
        evaluator = TBEvaluator(**config_evaluator)
        optimizer = TBOptimizer(**config_optimizer)
        model = TBModel(backbone=backbone,
                        readout=readout,
                        loss=loss,
                        optimizer=optimizer,
                        evaluator=evaluator,
                        compile=False)

        # train
        trainer = pl.Trainer(max_epochs=3,
                             accelerator="cpu",
                             enable_progress_bar=False,
                             log_every_n_steps=1)
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
        test_metrics = trainer.callback_metrics
        print('      Testing metrics\n', '-'*25)
        for key in test_metrics:
            print('{:<20s} {:>5.4f}'.format(key+':', test_metrics[key].item()))
