"""Preprocessor for datasets."""

import json
import os
import time

import torch
import torch_geometric
from filelock import FileLock
from torch_geometric.io import fs
from tqdm import tqdm

from topobench.data.utils import (
    ensure_serializable,
    load_inductive_splits,
    load_transductive_splits,
    make_hash,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


class PreProcessor(torch_geometric.data.InMemoryDataset):
    """Preprocessor for datasets.

    Parameters
    ----------
    dataset : list
        List of data objects.
    data_dir : str
        Path to the directory containing the data.
    transforms_config : DictConfig, optional
        Configuration parameters for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(self, dataset, data_dir, transforms_config=None, **kwargs):
        self.dataset = dataset
        self.preprocessing_time = 0
        if transforms_config is not None:
            self.transforms_applied = True
            pre_transform = self.instantiate_pre_transform(
                data_dir, transforms_config
            )

            # 1. Ensure the target directory exists so we can place a lock file in it
            os.makedirs(self.processed_data_dir, exist_ok=True)
            lock_path = os.path.join(
                self.processed_data_dir, "preprocessing.lock"
            )

            start_time = time.time()

            with FileLock(lock_path):
                # Attempt to recover from superset if data.pt is missing
                target_data_pt = os.path.join(self.processed_data_dir, "data.pt")
                if not os.path.exists(target_data_pt):
                    print(f"[PreProcessor] data.pt not found, attempting recovery...")
                    recovered = self._attempt_superset_recovery(data_dir, transforms_config)
                    if recovered:
                        # Save transform parameters immediately so next steps see correct metadata
                        self.save_transform_parameters()

                # When Process 1 finishes, Process 2 checks, sees data.pt, and skips.
                super().__init__(
                    self.processed_data_dir, None, pre_transform, **kwargs
                )
                self.save_transform_parameters()

            end_time = time.time()
            self.preprocessing_time = end_time - start_time

            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.load(self.processed_paths[0])
            self.data_list = [data for data in self]
        else:
            self.transforms_applied = False
            super().__init__(data_dir, None, None, **kwargs)
            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.data, self.slices = dataset._data, dataset.slices
            self.data_list = [data for data in dataset]

        # Some datasets have fixed splits, and those are stored as split_idx during loading
        # We need to store this information to be able to reproduce the splits afterwards
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx
        if hasattr(dataset, "split_idx_list"):
            self.split_idx_list = dataset.split_idx_list

    def _attempt_superset_recovery(self, data_dir, transforms_config):
        """Attempt to recover requested encodings from an existing superset run.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.
        """
        if not os.path.exists(data_dir):
            return False

        from topobench.utils.config_resolvers import _parse_encodings, get_all_encoding_dimensions
        
        # Identify which transforms in the CURRENT config are "recoverable"
        recoverable_keys = ["hopse_encoding", "CombinedPSEs", "CombinedFEs", "CombinedEncodings"]
        target_keys = [k for k in recoverable_keys if k in transforms_config]
        
        if not target_keys:
            return False

        print(f"[Superset Recovery] Target keys: {target_keys}")

        # Sibling search across all repo folders in data_dir
        for root, dirs, files in os.walk(data_dir):
            if "path_transform_parameters_dict.json" in files and "data.pt" in files:
                if root == self.processed_data_dir:
                    continue

                with open(os.path.join(root, "path_transform_parameters_dict.json"), "r") as f:
                    try:
                        sibling_params_dict = json.load(f)
                    except:
                        continue
                
                # Check if this sibling can satisfy ALL target keys
                sibling_mapping = {} # maps target_key -> sibling_key
                possible = True
                for tk in target_keys:
                    sk = None
                    if tk in sibling_params_dict:
                        sk = tk
                    elif tk == "hopse_encoding":
                        for alt in ["CombinedPSEs", "CombinedFEs", "CombinedEncodings"]:
                            if alt in sibling_params_dict:
                                sk = alt
                                break
                    elif tk in ["CombinedPSEs", "CombinedFEs", "CombinedEncodings"]:
                        if "hopse_encoding" in sibling_params_dict:
                            sk = "hopse_encoding"
                        elif tk == "CombinedPSEs" and "CombinedEncodings" in sibling_params_dict:
                            sk = "CombinedEncodings"
                        elif tk == "CombinedFEs" and "CombinedEncodings" in sibling_params_dict:
                            sk = "CombinedEncodings"
                    
                    if sk is None:
                        possible = False
                        break
                    
                    # Verify SK is a superset of TK
                    curr_p = self.transforms_parameters[tk]
                    sibl_p = sibling_params_dict[sk]
                    curr_encs = _parse_encodings(curr_p.get("encodings", []))
                    sibl_encs = _parse_encodings(sibl_p.get("encodings", []))
                    
                    if not all(enc in sibl_encs for enc in curr_encs):
                        possible = False
                        break
                    
                    # Parameters for the shared encodings must match (ignoring concat_to_x and device)
                    params_match = True
                    ignore_keys = ["concat_to_x", "device", "preprocessor_device"]
                    for enc in curr_encs:
                        curr_enc_p = dict(curr_p.get("parameters", {}).get(enc, {}))
                        sibl_enc_p = dict(sibl_p.get("parameters", {}).get(enc, {}))
                        for k in ignore_keys:
                            curr_enc_p.pop(k, None)
                            sibl_enc_p.pop(k, None)
                        
                        if curr_enc_p != sibl_enc_p:
                            params_match = False
                            break
                    if not params_match:
                        possible = False
                        break
                    
                    # Metadata check for hopse
                    if tk == "hopse_encoding" and sk == "hopse_encoding":
                        for k in ["neighborhoods", "max_rank", "copy_initial"]:
                            if curr_p.get(k) != sibl_p.get(k):
                                params_match = False
                                break
                    if not params_match:
                        possible = False
                        break
                        
                    sibling_mapping[tk] = sk
                
                if not possible:
                    continue

                print(f"[Superset Recovery] Found compatible cache at {root}")
                
                try:
                    processed_data = torch.load(os.path.join(root, "data.pt"), map_location="cpu")
                    
                    # Handle (data, slices) or data_list
                    data_obj = None
                    slices_obj = None
                    data_cls = None
                    
                    if isinstance(processed_data, tuple):
                        data_obj = processed_data[0]
                        slices_obj = processed_data[1]
                        if len(processed_data) > 2:
                            data_cls = processed_data[-1]
                    else:
                        data_obj = processed_data
                    
                    data_list = data_obj if isinstance(data_obj, list) else [data_obj]
                    
                    # Pre-calculate ALL sibling offsets for ANY transform that might have modified 'x'
                    # We assume 'x' concatenation order matches keys in sibling_params_dict
                    x_modifying_keys = ["CombinedFEs", "CombinedPSEs", "CombinedEncodings"]
                    sibling_x_offsets = {} # key -> (start_ptr, end_ptr)
                    total_ptr = 0
                    for k, p in sibling_params_dict.items():
                        if k in x_modifying_keys:
                            encs = _parse_encodings(p.get("encodings", []))
                            p_dict = p.get("parameters", {})
                            ptr_before = total_ptr
                            for enc in encs:
                                # ONLY add to ptr if it was actually concatenated to x!
                                if p_dict.get(enc, {}).get("concat_to_x", True):
                                    d = get_all_encoding_dimensions([enc], p_dict)[0]
                                    total_ptr += d
                            sibling_x_offsets[k] = (ptr_before, total_ptr)

                    # Now perform re-mapping for each target key
                    for data in data_list:
                        # Extract base_features_dim from the sibling's 'x'
                        x_attr = data['x'] if isinstance(data, dict) else data.x
                        base_features_dim = x_attr.shape[1] - total_ptr
                        base_x = x_attr[:, :base_features_dim]
                        
                        # Store temporary results to avoid overwriting while iterating
                        new_attrs = {} 
                        new_x_parts = [base_x]

                        # We MUST iterate through ALL transforms in the TARGET config
                        # to rebuild the new data object correctly.
                        for tk in transforms_config.keys():
                            if tk not in sibling_mapping:
                                # This transform was not recovered; if it's not a recoverable key, 
                                # we might have a problem because we are skipping process().
                                # For now, we assume only recoverable keys are used in these pipelines.
                                continue
                            
                            sk = sibling_mapping[tk]
                            curr_p = self.transforms_parameters[tk]
                            sibl_p = sibling_params_dict[sk]
                            curr_encs = _parse_encodings(curr_p.get("encodings", []))
                            sibl_encs = _parse_encodings(sibl_p.get("encodings", []))
                            
                            is_target_hopse = (tk == "hopse_encoding")
                            is_sibling_hopse = (sk == "hopse_encoding")

                            if is_target_hopse:
                                # Target is hopse: needs x{r}_{i}
                                if is_sibling_hopse:
                                    # Easy: direct re-mapping
                                    hop_off = 1 if sibl_p.get("copy_initial") else 0
                                    new_hop_off = 1 if curr_p.get("copy_initial") else 0
                                    if curr_p.get("copy_initial"):
                                        for r in range(curr_p.get("max_rank", 0) + 1):
                                            attr = f"x{r}_0"
                                            new_attrs[attr] = data[attr] if isinstance(data, dict) else getattr(data, attr)
                                    
                                    for r in range(curr_p.get("max_rank", 0) + 1):
                                        for new_idx, enc in enumerate(curr_encs):
                                            old_idx = sibl_encs.index(enc)
                                            old_attr, new_attr = f"x{r}_{old_idx + hop_off}", f"x{r}_{new_idx + new_hop_off}"
                                            new_attrs[new_attr] = data[old_attr] if isinstance(data, dict) else getattr(data, old_attr)
                                else:
                                    # Recover hopse from Combined* (x)
                                    start_off, _ = sibling_x_offsets[sk]
                                    ptr = start_off
                                    sibl_enc_offsets = {}
                                    for enc in sibl_encs:
                                        d = get_all_encoding_dimensions([enc], sibl_p.get("parameters", {}))[0]
                                        sibl_enc_offsets[enc] = (ptr, ptr + d)
                                        if sibl_p.get("parameters", {}).get(enc, {}).get("concat_to_x", True):
                                            ptr += d
                                    
                                    new_hop_off = 1 if curr_p.get("copy_initial") else 0
                                    if curr_p.get("copy_initial"):
                                        new_attrs["x0_0"] = base_x # best guess for x0_0
                                        # For higher ranks, x{r}_0 might not be available in Combined*
                                        
                                    for new_idx, enc in enumerate(curr_encs):
                                        s, e = sibl_enc_offsets[enc]
                                        val = x_attr[:, base_features_dim + s : base_features_dim + e]
                                        new_attrs[f"x0_{new_idx + new_hop_off}"] = val
                                        
                            else:
                                # Target is Combined*: needs concatenation in 'x' or separate attributes
                                if is_sibling_hopse:
                                    # Recover Combined* from hopse (x0_i)
                                    hop_off = 1 if sibl_p.get("copy_initial") else 0
                                    for enc in curr_encs:
                                        old_idx = sibl_encs.index(enc)
                                        val = data[f"x0_{old_idx + hop_off}"] if isinstance(data, dict) else getattr(data, f"x0_{old_idx + hop_off}")
                                        if curr_p.get("parameters", {}).get(enc, {}).get("concat_to_x", True):
                                            new_x_parts.append(val)
                                        else:
                                            new_attrs[enc] = val
                                else:
                                    # Recover Combined* from Combined* (x)
                                    start_off, _ = sibling_x_offsets[sk]
                                    ptr = start_off
                                    sibl_enc_offsets = {}
                                    for enc in sibl_encs:
                                        d = get_all_encoding_dimensions([enc], sibl_p.get("parameters", {}))[0]
                                        sibl_enc_offsets[enc] = (ptr, ptr + d)
                                        if sibl_p.get("parameters", {}).get(enc, {}).get("concat_to_x", True):
                                            ptr += d
                                            
                                    for enc in curr_encs:
                                        s, e = sibl_enc_offsets[enc]
                                        val = x_attr[:, base_features_dim + s : base_features_dim + e]
                                        if curr_p.get("parameters", {}).get(enc, {}).get("concat_to_x", True):
                                            new_x_parts.append(val)
                                        else:
                                            new_attrs[enc] = val
                        
                        # Apply all temporary results to data
                        if len(new_x_parts) > 1:
                            final_x = torch.cat(new_x_parts, dim=-1)
                            if isinstance(data, dict): data['x'] = final_x
                            else: data.x = final_x
                        else:
                            if isinstance(data, dict): data['x'] = base_x
                            else: data.x = base_x

                        for k, v in new_attrs.items():
                            if isinstance(data, dict): data[k] = v
                            else: setattr(data, k, v)
                    
                    data_obj, slices_obj = self.collate(data_list)
                    os.makedirs(self.processed_data_dir, exist_ok=True)
                    torch.save((data_obj, slices_obj) if data_cls is None else (data_obj, slices_obj, data_cls), 
                               os.path.join(self.processed_data_dir, "data.pt"))

                    print(f"[Superset Recovery] Successfully saved re-mapped data to {self.processed_data_dir}")
                    
                    # Update dim_all_encodings for hopse if needed
                    if "hopse_encoding" in self.transforms_parameters:
                        hp = self.transforms_parameters["hopse_encoding"]
                        hp["dim_all_encodings"] = get_all_encoding_dimensions(_parse_encodings(hp.get("encodings", [])), hp.get("parameters", {}))

                    return True
                    
                except Exception as e:
                    print(f"[Superset Recovery] Failed to recover from {root}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        return False

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return self.root

    @property
    def processed_file_names(self) -> str:
        """Return the name of the processed file.

        Returns
        -------
        str
            Name of the processed file.
        """
        return "data.pt"

    def instantiate_pre_transform(
        self, data_dir, transforms_config
    ) -> torch_geometric.transforms.Compose:
        """Instantiate the pre-transforms.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.

        Returns
        -------
        torch_geometric.transforms.Compose
            Pre-transform object.
        """
        from torch_geometric.transforms import ToDevice

        if transforms_config.keys() == {"liftings"}:
            transforms_config = transforms_config.liftings

        if "transform_name" in transforms_config:
            config_items = [
                (transforms_config.transform_name, transforms_config)
            ]
        else:
            config_items = transforms_config.items()

        pre_transforms_list = []
        pre_transforms_dict = {}

        # Track where the graph currently lives in the pipeline
        current_device = "cpu"

        for key, value in config_items:
            kwargs = dict(value)

            requested_device = kwargs.pop("preprocessor_device", "cpu")

            target_device = (
                "cuda"
                if requested_device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )

            transform_instance = DataTransform(**kwargs)
            pre_transforms_dict[key] = transform_instance

            if target_device != current_device:
                pre_transforms_list.append(ToDevice(target_device))
                current_device = target_device

            pre_transforms_list.append(transform_instance)

        # If the pipeline ends while the graph is still on the GPU,
        # we MUST pull it back to the CPU before PyTorch Geometric saves it to disk.
        if current_device == "cuda":
            pre_transforms_list.append(ToDevice("cpu"))

        pre_transforms = torch_geometric.transforms.Compose(
            pre_transforms_list
        )

        self.set_processed_data_dir(
            pre_transforms_dict, data_dir, transforms_config
        )
        return pre_transforms

    def set_processed_data_dir(
        self, pre_transforms_dict, data_dir, transforms_config
    ) -> None:
        """Set the processed data directory.

        Parameters
        ----------
        pre_transforms_dict : dict
            Dictionary containing the pre-transforms.
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.
        """
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(
            *[data_dir, repo_name, f"{params_hash}"]
        )

    def save_transform_parameters(self) -> None:
        """Save the transform parameters."""
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f, indent=4)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters) as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                # Always overwrite metadata during recovery or if there's a forced change
                with open(path_transform_parameters, "w") as f:
                    json.dump(self.transforms_parameters, f, indent=4)

            print(
                f"Transform parameters synced at: {self.processed_data_dir}"
            )

    def process(self) -> None:
        """Method that processes the data."""
        if isinstance(
            self.dataset,
            (torch_geometric.data.Dataset, torch.utils.data.Dataset),
        ):
            data_list = [data for data in self.dataset]
        elif isinstance(self.dataset, torch_geometric.data.Data):
            data_list = [self.dataset]

        if self.pre_transform is not None:
            print(f"\nApplying transforms to {len(data_list)} graphs...")
            self.data_list = [
                self.pre_transform(d)
                for d in tqdm(
                    data_list, desc="Processing graphs", unit="graph"
                )
            ]
        else:
            self.data_list = data_list

        self._data, self.slices = self.collate(self.data_list)
        self._data_list = None  # Reset cache.

        assert isinstance(self._data, torch_geometric.data.Data)
        self.save(self.data_list, self.processed_paths[0])

    def load(self, path: str) -> None:
        r"""Load the dataset from the file path `path`.

        Parameters
        ----------
        path : str
            The path to the processed data.
        """
        out = fs.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) >= 2 and len(out) <= 4
        if len(out) == 2:  # Backward compatibility (1).
            data, self.slices = out
        elif len(out) == 3:  # Backward compatibility (2).
            data, self.slices, data_cls = out
        else:  # TU Datasets store additional element (__class__) in the processed file
            data, self.slices, sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def load_dataset_splits(
        self, split_params
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """Load the dataset splits.

        Parameters
        ----------
        split_params : dict
            Parameters for loading the dataset splits.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test datasets.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting == "inductive":
            return load_inductive_splits(self, split_params)
        elif split_params.learning_setting == "transductive":
            return load_transductive_splits(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting.\
                Please define either 'inductive' or 'transductive'."
            )
