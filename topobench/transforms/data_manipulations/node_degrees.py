"""Node degrees transform."""

import torch_geometric


class NodeDegrees(torch_geometric.transforms.BaseTransform):
    r"""A transform that calculates the node degrees of the input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "node_degrees"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        field_to_process = [
            key
            for key in data.to_dict()
            for field_substring in self.parameters["selected_fields"]
            if field_substring in key and key != "incidence_0"
        ]
        for field in field_to_process:
            data = self.calculate_node_degrees(data, field)

        return data

    def calculate_node_degrees(
        self, data: torch_geometric.data.Data, field: str
    ) -> torch_geometric.data.Data:
        r"""Calculate the node degrees of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        field : str
            The field to calculate the node degrees.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        if data[field].is_sparse:
            degrees = abs(data[field].to_dense()).sum(1)
        else:
            assert field == "edge_index", (
                "Following logic of finding degrees is only implemented for edge_index"
            )

            # Get number of nodes
            if data.get("num_nodes", None):
                max_num_nodes = data["num_nodes"]
            else:
                max_num_nodes = data["x"].shape[0]
            degrees = (
                torch_geometric.utils.to_dense_adj(
                    data[field],
                    max_num_nodes=max_num_nodes,
                )
                .squeeze(0)
                .sum(1)
            )

        if "incidence" in field:
            field_name = (
                str(int(field.split("_")[1]) - 1) + "_cell" + "_degrees"
            )
        else:
            field_name = "node_degrees"

        data[field_name] = degrees.unsqueeze(1)
        return data


# """Node degrees transform."""

# import torch_geometric


# class NodeDegrees(torch_geometric.transforms.BaseTransform):
#     r"""A transform that calculates the node degrees of the input graph.

#     Parameters
#     ----------
#     **kwargs : optional
#         Parameters for the base transform.
#     """

#     def __init__(self, **kwargs):
#         super().__init__()
#         self.type = "node_degrees"
#         self.parameters = kwargs

#         # Check that in case self.parameters['selected_fields'] consists of incidences then there is a list of upper/lower degrees
#         temp = []
#         for field in self.parameters["selected_fields"]:
#             if "incidence" in field:
#                 assert self.parameters["degrees_types"] is not None, (
#                     "If incidence fields are selected, then degrees_types must be provided"
#                 )

#                 temp.append(field)

#         # Check that the number of selected fields is equal to the number of degrees types

#         assert len(self.parameters["selected_fields"]) == len(
#             self.parameters["degrees_types"]
#         ), (
#             "The number of selected_fields fields must be equal to the number of degrees_types"
#         )

#         # Check that the number of stat_var is equal to the number of degrees
#         assert len(self.parameters["selected_fields"]) == len(
#             self.parameters["stat_vars"]
#         ), (
#             "The number of selected_fields fields must be equal to the number of stat_var"
#         )

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

#     def forward(self, data: torch_geometric.data.Data):
#         r"""Apply the transform to the input data.

#         Parameters
#         ----------
#         data : torch_geometric.data.Data
#             The input data.

#         Returns
#         -------
#         torch_geometric.data.Data
#             The transformed data.
#         """
#         field_to_process = [
#             key
#             for key in data.to_dict()
#             for field_substring in self.parameters["selected_fields"]
#             if field_substring in key and key != "incidence_0"
#         ]
#         assert len(field_to_process) == len(
#             self.parameters["degrees_types"]
#         ), (
#             "The number of selected fields must be equal to the number of degrees types"
#         )

#         for field, degree_type, stat_var in zip(
#             field_to_process,
#             self.parameters["degrees_types"],
#             self.parameters["stat_vars"],
#         ):
#             data = self.calculate_node_degrees(
#                 data, field, degree_type, stat_var
#             )

#         return data

#     def calculate_node_degrees(
#         self,
#         data: torch_geometric.data.Data,
#         field: str,
#         degree_type: str,
#         stat_var: str = None,
#     ) -> torch_geometric.data.Data:
#         r"""Calculate the node degrees of the input data.

#         Parameters
#         ----------
#         data : torch_geometric.data.Data
#             The input data.
#         field : str
#             The field to calculate the node degrees.
#         degree_type : str
#             The type of degree to calculate (e.g., "upper", "lower").
#         stat_var : str, optional
#             The variable name to use for the degree saving (default is None). If not given, the field name will identified automatically.

#         Returns
#         -------
#         torch_geometric.data.Data
#             The transformed data.
#         """
#         if data[field].is_sparse:
#             dim_to_sum = (
#                 1 if degree_type == "up_cell_degree" else 0
#             )  # dim_to_sum
#             degrees = abs(data[field].to_dense()).sum(dim_to_sum)
#         else:
#             assert field == "edge_index", (
#                 "Following logic of finding degrees is only implemented for edge_index"
#             )
#             assert degree_type == "up_cell_degree", (
#                 "finding edge degrees for nodes is equal to finding up cell degrees."
#             )
#             # Get number of nodes
#             if data.get("num_nodes", None):
#                 max_num_nodes = data["num_nodes"]
#             else:
#                 max_num_nodes = data["x"].shape[0]
#             degrees = (
#                 torch_geometric.utils.to_dense_adj(
#                     data[field],
#                     max_num_nodes=max_num_nodes,
#                 )
#                 .squeeze(0)
#                 .sum(1)
#             )

#         if stat_var is not None:
#             field_name = stat_var
#         else:
#             if "incidence" in field:
#                 field_name = (
#                     str(int(field.split("_")[1]) - 1) + "_cell" + "_degrees"
#                 )
#             else:
#                 field_name = "node_degrees"

#         data[field_name] = degrees.unsqueeze(1)
#         return data
