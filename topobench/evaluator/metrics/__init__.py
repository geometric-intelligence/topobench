"""Init file for custom metrics in evaluator module."""

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any


class LoadManager:
    """Manages automatic discovery and registration of loss classes."""

    @staticmethod
    def is_metric_class(obj: Any) -> bool:
        """Check if an object is a valid metric class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid loss class.

        Returns
        -------
        bool
            True if the object is a valid loss class (non-private class
            with 'FeatureEncoder' in name), False otherwise.
        """
        try:
            from torchmetrics import Metric

            return (
                inspect.isclass(obj)
                and not obj.__name__.startswith("_")
                and issubclass(obj, Metric)
                and obj is not Metric
            )
        except ImportError:
            return False

    @classmethod
    def discover_metrics(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all metric classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        Dict[str, Type]
            Dictionary mapping loss class names to their corresponding class objects.
        """
        metrics = {}
        package_dir = Path(package_path).parent

        # Add parent directory to sys.path to ensure imports work
        parent_dir = str(package_dir.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Iterate through all .py files in the directory
        for file_path in package_dir.glob("*.py"):
            if file_path.stem == "__init__":
                continue

            try:
                # Use importlib to safely import the module
                package_name = package_dir.parent.parent.stem + "." + package_dir.parent.stem + "." + package_dir.stem
                module_name = f"{package_name}.{file_path.stem}"
                module = importlib.import_module(module_name)

                # Find all loss classes in the module
                new_metrics = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        cls.is_metric_class(obj)
                        and obj.__module__ == module.__name__
                    )
                }
                metrics.update(new_metrics)
            except ImportError as e:
                print(f"Could not import module {module_name}: {e}")

        return metrics


# Dynamically create the loss manager and discover losses
manager = LoadManager()
CUSTOM_METRICS = manager.discover_metrics(__file__)
CUSTOM_METRICS_list = list(CUSTOM_METRICS.keys())

# Combine manual and discovered losses
all_metrics = {**CUSTOM_METRICS}

# Generate __all__
__all__ = [
    "CUSTOM_METRICS",
    "CUSTOM_METRICS_list",
    *list(all_metrics.keys()),
]

# Update locals for direct import
locals().update(all_metrics)

# from .example import ExampleRegressionMetric

# __all__ = [
#     "ExampleRegressionMetric",
# ]
