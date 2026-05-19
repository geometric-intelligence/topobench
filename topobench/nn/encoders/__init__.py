"""Init file for encoder module with automated encoder discovery."""

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any


class LoadManager:
    """Manages automatic discovery and registration of encoder classes."""

    @staticmethod
    def is_encoder_class(obj: Any) -> bool:
        """Check if an object is a valid encoder class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid encoder class.

        Returns
        -------
        bool
            True if the object is a valid encoder class (non-private class
            with 'FeatureEncoder' in name), False otherwise.
        """
        try:
            from .base import AbstractFeatureEncoder

            return (
                inspect.isclass(obj)
                and not obj.__name__.startswith("_")
                and issubclass(obj, AbstractFeatureEncoder)
                and obj is not AbstractFeatureEncoder
            )
        except ImportError:
            return False

    @classmethod
    def discover_encoders(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all encoder classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        Dict[str, Type]
            Dictionary mapping encoder class names to their corresponding class objects.
        """
        encoders = {}
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

                # Find all encoder classes in the module
                new_encoders = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        cls.is_encoder_class(obj)
                        and obj.__module__ == module.__name__
                    )
                }
                encoders.update(new_encoders)
            except ImportError as e:
                print(f"Could not import module {module_name}: {e}")

        return encoders


# Dynamically create the encoder manager and discover encoders
manager = LoadManager()
FEATURE_ENCODERS = manager.discover_encoders(__file__)
FEATURE_ENCODERS_list = list(FEATURE_ENCODERS.keys())


# Combine manual and discovered encoders
all_encoders = {**FEATURE_ENCODERS}

# Generate __all__
__all__ = [
    "FEATURE_ENCODERS",
    "FEATURE_ENCODERS_list",
    *list(all_encoders.keys()),
]

# Update locals for direct import
locals().update(all_encoders)
