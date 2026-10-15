"""Combinatorial backbones with automated exports."""

import inspect
from importlib import import_module
from pathlib import Path
from typing import Any


class BackboneExportsManager:
    """Manages automatic discovery and registration of backbone classes."""

    @staticmethod
    def is_backbone_class(obj: Any) -> bool:
        """Check if an object is a valid backbone class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid backbone class.

        Returns
        -------
        bool
            True if the object is a valid backbone class (non-private class defined in __main__), False otherwise.
        """
        return (
            inspect.isclass(obj)
            and obj.__module__ == "__main__"
            and not obj.__name__.startswith("_")
        )

    @classmethod
    def discover_backbones(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all backbone classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        dict[str, type]
            Dictionary mapping class names to their corresponding class objects.
        """
        backbones = {}
        package_dir = Path(package_path).parent

        for file_path in package_dir.glob("*.py"):
            if file_path.stem == "__init__":
                continue

            # Import under the package name so classes keep their identity,
            # and export only a module's ``__all__`` when it declares one.
            module = import_module(f"{__name__}.{file_path.stem}")
            new_backbones = {
                name: obj
                for name, obj in inspect.getmembers(module)
                if inspect.isclass(obj)
                and obj.__module__ == module.__name__
                and not name.startswith("_")
                and name in getattr(module, "__all__", [name])
            }
            backbones.update(new_backbones)
        return backbones


# Create the exports manager
manager = BackboneExportsManager()

# Automatically discover and populate BACKBONE_CLASSES
BACKBONE_CLASSES = manager.discover_backbones(__file__)

# Automatically generate __all__
__all__ = [*BACKBONE_CLASSES.keys(), "BACKBONE_CLASSES"]

# For backwards compatibility, also create individual imports
locals().update(BACKBONE_CLASSES)
