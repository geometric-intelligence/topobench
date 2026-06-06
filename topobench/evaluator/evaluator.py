"""This module contains the Evaluator class that is responsible for computing the metrics."""

from torchmetrics import MetricCollection

from topobench.evaluator import METRICS, AbstractEvaluator


class TBEvaluator(AbstractEvaluator):
    r"""Evaluator class that is responsible for computing the metrics.

    Parameters
    ----------
    task : str
        The task type. It can be "classification", "regression", "multilabel classification",
        or "multioutput classification".
    **kwargs : dict
        Additional arguments for the class. The arguments depend on the task.
        In "classification" scenario, the following arguments are expected:
        - num_classes (int): The number of classes.
        - metrics (list[str]): A list of classification metrics to be computed.
        In "regression" scenario, the following arguments are expected:
        - metrics (list[str]): A list of regression metrics to be computed.
        In "multi_output_classification" scenario (e.g., Betti numbers), the following are expected:
        - num_output_dimbers (list[int]): List of number of classes for each output.
        - metrics (list[str]): A list of metrics with suffixes (e.g., "accuracy_0", "f1_1").
    """

    def __init__(self, task, **kwargs):
        # Define the task
        self.task = task

        # Define the metrics depending on the task
        if self.task == "classification":
            num_classes = kwargs.get("num_classes", 1)
            parameters = {"num_classes": num_classes}
            parameters["task"] = "multiclass"
            metric_names = kwargs["metrics"]

        elif self.task == "multilabel classification":
            num_classes = kwargs.get("num_classes", 1)
            parameters = {"num_classes": num_classes}
            parameters["task"] = "multilabel"
            parameters["num_labels"] = num_classes
            metric_names = kwargs["metrics"]

        elif self.task == "multioutput classification":
            # Handle multi-output classification (e.g., Betti numbers)
            # Each output is treated as a separate classification task
            self.multioutput_classes = kwargs["multioutput_classes"]
            metric_names = kwargs["metrics"]

            # Build parameters for each output
            parameters_multioutput = {}
            for output_dim, output_classes in enumerate(
                self.multioutput_classes
            ):
                out_dict = {}
                out_dict["task"] = "multiclass"
                out_dict["num_classes"] = output_classes
                parameters_multioutput[output_dim] = out_dict

        elif self.task == "regression":
            parameters = {}
            metric_names = kwargs["metrics"]

        else:
            raise ValueError(f"Invalid task {task}")

        metrics = {}
        for name in metric_names:
            metric_id = name.split("-")[0]
            parameters = (
                parameters
                if self.task != "multioutput classification"
                else parameters_multioutput[int(name.split("-")[1])]
            )
            if metric_id in ["recall", "precision", "auroc", "f1", "f1_macro"]:
                metrics[name] = METRICS[metric_id](
                    average="macro", **parameters
                )
            elif metric_id == "f1_weighted":
                metrics[name] = METRICS[metric_id](
                    average="weighted", **parameters
                )
            elif metric_id == "confusion_matrix":
                metrics[name] = METRICS[metric_id](**parameters)
            elif metric_id == "rmse":
                # RMSE is MSE with squared=False
                metrics[name] = METRICS[metric_id](squared=False, **parameters)
            else:
                metrics[name] = METRICS[metric_id](**parameters)
        self.metrics = MetricCollection(metrics)

        self.best_metric = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, metrics={self.metrics})"

    def update(self, model_out: dict):
        r"""Update the metrics with the model output.

        Parameters
        ----------
        model_out : dict
            The model output. It should contain the following keys:
            - logits : torch.Tensor
            The model predictions.
            - labels : torch.Tensor
            The ground truth labels.
            - batch : torch_geometric.data.Data (optional)
            The batch data containing target normalizer stats.

        Raises
        ------
        ValueError
            If the task is not valid.
        """
        preds = model_out["logits"].cpu()
        target = model_out["labels"].cpu()

        if self.task == "regression":
            self.metrics.update(preds, target.unsqueeze(1))

        elif self.task == "classification":
            self.metrics.update(preds, target)

        elif self.task == "multioutput classification":
            # Handle multi-output classification (e.g., Betti numbers)
            # Round predictions to nearest integer and clamp to valid range
            preds = preds.detach().clone().round().long()

            for metric_name in self.metrics:
                _, dim = metric_name.split("-")
                dim_idx = int(dim)

                # Clamp predictions to valid range for this output
                current_preds = preds[:, dim_idx].clamp(
                    min=0, max=self.multioutput_classes[dim_idx] - 1
                )

                self.metrics[metric_name].update(
                    current_preds, target[:, dim_idx]
                )

        elif self.task == "multilabel classification":
            self.metrics.update(preds, target)
            # Raise not supported error
            raise NotImplementedError(
                "Multilabel classification evaluator is not supported yet"
            )

    def compute(self):
        r"""Compute the metrics.

        Returns
        -------
        dict
            Dictionary containing the computed metrics.
        """
        return self.metrics.compute()

    def reset(self):
        """Reset the metrics.

        This method should be called after each epoch.
        """
        self.metrics.reset()
