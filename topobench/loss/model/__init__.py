"""This module implements the loss functions for the topobench package."""

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any


class LoadManager:
    """Manages automatic discovery and registration of loss classes."""

    @staticmethod
    def is_encoder_class(obj: Any) -> bool:
        """Check if an object is a valid loss class.

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
            from ..base import AbstractLoss

            return (
                inspect.isclass(obj)
                and not obj.__name__.startswith("_")
                and issubclass(obj, AbstractLoss)
                and obj is not AbstractLoss
            )
        except ImportError:
            return False

    @classmethod
    def discover_losses(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all loss classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        Dict[str, Type]
            Dictionary mapping loss class names to their corresponding class objects.
        """
        losses = {}
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
                new_losses = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        cls.is_encoder_class(obj)
                        and obj.__module__ == module.__name__
                    )
                }
                losses.update(new_losses)
            except ImportError as e:
                print(f"Could not import module {module_name}: {e}")

        return losses


# Dynamically create the loss manager and discover losses
manager = LoadManager()
LOSSES = manager.discover_losses(__file__)
LOSSES_list = list(LOSSES.keys())

# Combine manual and discovered losses
all_encoders = {**LOSSES}

# Generate __all__
__all__ = [
    "LOSSES",
    "LOSSES_list",
    *list(all_encoders.keys()),
]

# Update locals for direct import
locals().update(all_encoders)


# """Init file for custom loss module."""

# from .GraphMLPLoss import GraphMLPLoss
# from .DGMLoss import DGMLoss

# __all__ = [
#     "GraphMLPLoss",
#     "DGMLoss",
# ]
