"""Some models implemented for TopoBenchX with automated exports."""

import inspect
from importlib import import_module
from pathlib import Path
from typing import Any


class ModelExportsManager:
    """Manages automatic discovery and registration of model classes."""

    @staticmethod
    def is_model_class(obj: Any) -> bool:
        """Check if an object is a valid model class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid model class.

        Returns
        -------
        bool
            True if the object is a valid model class (non-private class defined in __main__), False otherwise.
        """
        return (
            inspect.isclass(obj)
            and obj.__module__ == "__main__"
            and not obj.__name__.startswith("_")
        )

    @classmethod
    def discover_models(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all model classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        dict[str, type]
            Dictionary mapping class names to their corresponding class objects.
        """
        models = {}
        package_dir = Path(package_path).parent

        for subpackage in package_dir.iterdir():
            if subpackage.is_dir() and (subpackage / "__init__.py").exists():
                for file_path in subpackage.glob("*.py"):
                    if file_path.stem == "__init__":
                        continue

                    module_name = (
                        f"{__name__}.{subpackage.stem}.{file_path.stem}"
                    )
                    module = import_module(module_name)
                    new_models = {
                        name: obj
                        for name, obj in inspect.getmembers(module)
                        if inspect.isclass(obj)
                        and obj.__module__ == module.__name__
                        and not name.startswith("_")
                        and name in getattr(module, "__all__", [name])
                    }
                    models.update(new_models)
        return models


# Create the exports manager
manager = ModelExportsManager()

# Automatically discover and populate MODEL_CLASSES
MODEL_CLASSES = manager.discover_models(__file__)

# Automatically generate __all__
__all__ = [*MODEL_CLASSES.keys(), "MODEL_CLASSES"]

# For backwards compatibility, also create individual imports
locals().update(MODEL_CLASSES)
