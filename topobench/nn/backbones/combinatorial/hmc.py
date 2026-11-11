"""HMC: Hierarchical Message-passing Combinatorial Complex Attention Network.

Implementation of the combinatorial complex attention neural network
(CCANN, also known as HMC) proposed in "Topological Deep Learning: Going
Beyond Graph Data" (Hajij et al., 2023, https://arxiv.org/abs/2206.00606),
cross-checked against the reference implementation in TopoModelX
(https://github.com/pyt-team/TopoModelX,
``topomodelx/nn/combinatorial/hmc.py``).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


def sparse_row_norm(sparse_tensor):
    r"""Normalize a sparse tensor by dividing each row by its sum.

    Rows whose sum is zero are left unnormalized (divided by one) to
    avoid producing non-finite values.

    Parameters
    ----------
    sparse_tensor : torch.sparse.Tensor
        Sparse tensor of shape ``[n_rows, n_columns]``.

    Returns
    -------
    torch.sparse.Tensor
        Row-normalized sparse tensor with the same shape as the input.
    """
    sparse_tensor = sparse_tensor.coalesce()
    row_sum = torch.sparse.sum(sparse_tensor, dim=1).to_dense()
    row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    values = sparse_tensor.values() / row_sum[sparse_tensor.indices()[0]]
    return torch.sparse_coo_tensor(
        sparse_tensor.indices(), values, sparse_tensor.shape
    ).coalesce()


class HBS(nn.Module):
    r"""Higher Order Attention Block for squared neighborhood matrices.

    Given a neighborhood matrix :math:`N` mapping the :math:`s`-th
    skeleton of a combinatorial complex to itself, the block computes
    (Definitions 31 and 32 of Hajij et al. 2023,
    https://arxiv.org/abs/2206.00606):

    .. math::
        \text{HBS}_N(X) =
            \phi\left(\sum_{p=1}^{m}(N^p \odot A_p) X W_p\right),

    where :math:`W_p` are learnable weight matrices, :math:`\odot` is
    the Hadamard product, :math:`\phi` an optional activation, and
    :math:`A_p` attention matrices with entries

    .. math::
        A_p(i, j) = S(\text{LeakyReLU}([X_i W_p \| X_j W_p] a_p)),

    where :math:`a_p` is a learnable attention vector and :math:`S` a
    row-wise normalization (softmax or plain row normalization).

    Parameters
    ----------
    source_in_channels : int
        Number of input features for the source cells.
    source_out_channels : int
        Number of output features for the source cells.
    negative_slope : float, optional
        Negative slope of the LeakyReLU activation (default: 0.2).
    softmax : bool, optional
        Whether to use softmax or row normalization for the attention
        matrix (default: False).
    m_hop : int, optional
        Maximum number of hops considered (default: 1).
    update_func : str, optional
        Activation function :math:`\phi`, one of ``None``,
        ``"sigmoid"``, ``"relu"``, ``"tanh"`` (default: None).
    initialization : str, optional
        Weight initialization method, either ``"xavier_uniform"`` or
        ``"xavier_normal"`` (default: ``"xavier_uniform"``).
    """

    def __init__(
        self,
        source_in_channels,
        source_out_channels,
        negative_slope=0.2,
        softmax=False,
        m_hop=1,
        update_func=None,
        initialization="xavier_uniform",
    ):
        super().__init__()
        self.source_in_channels = source_in_channels
        self.source_out_channels = source_out_channels
        self.m_hop = m_hop
        self.update_func = update_func
        self.initialization = initialization
        self.negative_slope = negative_slope
        self.softmax = softmax

        self.weight = nn.ParameterList(
            [
                Parameter(torch.empty(source_in_channels, source_out_channels))
                for _ in range(m_hop)
            ]
        )
        self.att_weight = nn.ParameterList(
            [
                Parameter(torch.empty(2 * source_out_channels, 1))
                for _ in range(m_hop)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float, optional
            Gain for the weight initialization (default: 1.414).
        """
        if self.initialization == "xavier_uniform":
            init_fn = nn.init.xavier_uniform_
        elif self.initialization == "xavier_normal":
            init_fn = nn.init.xavier_normal_
        else:
            raise ValueError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )
        for w, a in zip(self.weight, self.att_weight, strict=True):
            init_fn(w, gain=gain)
            init_fn(a.view(-1, 1), gain=gain)

    def update(self, message):
        r"""Apply the activation function :math:`\phi` to the message.

        Parameters
        ----------
        message : torch.Tensor
            Message tensor of shape ``[n_cells, out_channels]``.

        Returns
        -------
        torch.Tensor
            Activated message tensor.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(message)
        if self.update_func == "relu":
            return F.relu(message)
        if self.update_func == "tanh":
            return torch.tanh(message)
        raise ValueError(
            "Update function not recognized. "
            "Should be either sigmoid, relu or tanh."
        )

    def attention(self, message, a_p, neighborhood_p):
        r"""Compute the attention matrix :math:`A_p`.

        Parameters
        ----------
        message : torch.Tensor
            Message tensor :math:`X W_p` of shape
            ``[n_cells, source_out_channels]``.
        a_p : torch.Tensor
            Learnable attention vector of shape
            ``[2 * source_out_channels, 1]``.
        neighborhood_p : torch.sparse.Tensor
            Neighborhood matrix :math:`N^p` of shape
            ``[n_cells, n_cells]``.

        Returns
        -------
        torch.sparse.Tensor
            Attention matrix of shape ``[n_cells, n_cells]``.
        """
        n_cells = message.shape[0]
        index_i, index_j = neighborhood_p.indices()
        s_to_s = torch.cat([message[index_i], message[index_j]], dim=1)
        e_p = torch.sparse_coo_tensor(
            indices=neighborhood_p.indices(),
            values=F.leaky_relu(
                torch.matmul(s_to_s, a_p),
                negative_slope=self.negative_slope,
            ).squeeze(1),
            size=(n_cells, n_cells),
            device=message.device,
        )
        if self.softmax:
            return torch.sparse.softmax(e_p, dim=1)
        return sparse_row_norm(e_p)

    def forward(self, x_source, neighborhood):
        r"""Compute the forward pass.

        Parameters
        ----------
        x_source : torch.Tensor
            Input features of shape ``[n_cells, source_in_channels]``.
        neighborhood : torch.sparse.Tensor
            Neighborhood matrix :math:`N` of shape
            ``[n_cells, n_cells]``.

        Returns
        -------
        torch.Tensor
            Output features of shape ``[n_cells, source_out_channels]``.
        """
        messages = [torch.mm(x_source, w) for w in self.weight]

        neighborhood_p = torch.sparse_coo_tensor(
            torch.arange(x_source.shape[0], device=x_source.device).repeat(
                2, 1
            ),
            torch.ones(x_source.shape[0], device=x_source.device),
            (x_source.shape[0], x_source.shape[0]),
        )
        m_hop_matrices = []
        for _ in range(self.m_hop):
            neighborhood_p = torch.sparse.mm(
                neighborhood_p, neighborhood
            ).coalesce()
            m_hop_matrices.append(neighborhood_p)

        result = torch.zeros(
            x_source.shape[0],
            self.source_out_channels,
            device=x_source.device,
        )
        for message, n_p, a_p in zip(
            messages, m_hop_matrices, self.att_weight, strict=True
        ):
            att_p = self.attention(message, a_p, n_p)
            n_p_att = torch.sparse_coo_tensor(
                indices=n_p.indices(),
                values=att_p.values() * n_p.values(),
                size=n_p.shape,
                device=message.device,
            )
            result = result + torch.mm(n_p_att, message)

        if self.update_func is None:
            return result
        return self.update(result)


class HBNS(nn.Module):
    r"""Higher Order Attention Block for non-squared neighborhood matrices.

    Given a neighborhood matrix :math:`N` mapping the :math:`s`-th
    skeleton of a combinatorial complex to its :math:`t`-th skeleton,
    with :math:`s \neq t`, the block computes (Definitions 31 and 33 of
    Hajij et al. 2023, https://arxiv.org/abs/2206.00606):

    .. math::
        \text{HBNS}_N(X_s, X_t) = (Y_s, Y_t), \quad
        Y_s = \phi((N^T \odot A_t) X_t W_t), \quad
        Y_t = \phi((N \odot A_s) X_s W_s),

    where :math:`W_s, W_t` are learnable weight matrices and
    :math:`A_s, A_t` attention matrices computed from the learnable
    attention vector :math:`a` as in Eqs. (12)-(13) of the paper:

    .. math::
        e_{ij} &= \text{LeakyReLU}(
            [(X_s)_j W_s \| (X_t)_i W_t] a), \\
        f_{ji} &= \text{LeakyReLU}(
            [(X_t)_i W_t \| (X_s)_j W_s] \operatorname{rev}(a)),

    followed by row-wise normalization of :math:`e` and :math:`f`
    (softmax or division by the row sum). Here
    :math:`\operatorname{rev}(a)` swaps the source and target portions of
    the attention vector, matching Definition 33.

    Parameters
    ----------
    source_in_channels : int
        Number of input features for the source cells.
    source_out_channels : int
        Number of output features for the source cells.
    target_in_channels : int
        Number of input features for the target cells.
    target_out_channels : int
        Number of output features for the target cells.
    negative_slope : float, optional
        Negative slope of the LeakyReLU activation (default: 0.2).
    softmax : bool, optional
        Whether to use softmax or row normalization for the attention
        matrices (default: False).
    update_func : str, optional
        Activation function :math:`\phi`, one of ``None``,
        ``"sigmoid"``, ``"relu"``, ``"tanh"`` (default: None).
    initialization : str, optional
        Weight initialization method, either ``"xavier_uniform"`` or
        ``"xavier_normal"`` (default: ``"xavier_uniform"``).
    """

    def __init__(
        self,
        source_in_channels,
        source_out_channels,
        target_in_channels,
        target_out_channels,
        negative_slope=0.2,
        softmax=False,
        update_func=None,
        initialization="xavier_uniform",
    ):
        super().__init__()
        self.source_in_channels = source_in_channels
        self.source_out_channels = source_out_channels
        self.target_in_channels = target_in_channels
        self.target_out_channels = target_out_channels
        self.update_func = update_func
        self.initialization = initialization
        self.negative_slope = negative_slope
        self.softmax = softmax

        self.w_s = Parameter(
            torch.empty(source_in_channels, target_out_channels)
        )
        self.w_t = Parameter(
            torch.empty(target_in_channels, source_out_channels)
        )
        self.att_weight = Parameter(
            torch.empty(target_out_channels + source_out_channels, 1)
        )
        self.reset_parameters()

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float, optional
            Gain for the weight initialization (default: 1.414).
        """
        if self.initialization == "xavier_uniform":
            init_fn = nn.init.xavier_uniform_
        elif self.initialization == "xavier_normal":
            init_fn = nn.init.xavier_normal_
        else:
            raise ValueError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )
        init_fn(self.w_s, gain=gain)
        init_fn(self.w_t, gain=gain)
        init_fn(self.att_weight.view(-1, 1), gain=gain)

    def update(self, message_on_source, message_on_target):
        r"""Apply the activation function :math:`\phi` to the messages.

        Parameters
        ----------
        message_on_source : torch.Tensor
            Message tensor for the source cells of shape
            ``[n_source_cells, source_out_channels]``.
        message_on_target : torch.Tensor
            Message tensor for the target cells of shape
            ``[n_target_cells, target_out_channels]``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Activated source and target message tensors.
        """
        if self.update_func == "sigmoid":
            return (
                torch.sigmoid(message_on_source),
                torch.sigmoid(message_on_target),
            )
        if self.update_func == "relu":
            return F.relu(message_on_source), F.relu(message_on_target)
        if self.update_func == "tanh":
            return torch.tanh(message_on_source), torch.tanh(message_on_target)
        raise ValueError(
            "Update function not recognized. "
            "Should be either sigmoid, relu or tanh."
        )

    def attention(self, s_message, t_message, neighborhood):
        r"""Compute the attention matrices :math:`A_s` and :math:`A_t`.

        Parameters
        ----------
        s_message : torch.Tensor
            Source message tensor :math:`X_s W_s` of shape
            ``[n_source_cells, target_out_channels]``.
        t_message : torch.Tensor
            Target message tensor :math:`X_t W_t` of shape
            ``[n_target_cells, source_out_channels]``.
        neighborhood : torch.sparse.Tensor
            Coalesced neighborhood matrix :math:`N` of shape
            ``[n_target_cells, n_source_cells]``.

        Returns
        -------
        tuple of (torch.sparse.Tensor, torch.sparse.Tensor)
            Attention matrices :math:`A_s` (shape
            ``[n_target_cells, n_source_cells]``) and :math:`A_t`
            (shape ``[n_source_cells, n_target_cells]``).
        """
        target_indices, source_indices = neighborhood.indices()

        s_to_t = torch.cat(
            [s_message[source_indices], t_message[target_indices]], dim=1
        )
        t_to_s = torch.cat(
            [t_message[target_indices], s_message[source_indices]], dim=1
        )

        e = torch.sparse_coo_tensor(
            indices=torch.stack([target_indices, source_indices]),
            values=F.leaky_relu(
                torch.matmul(s_to_t, self.att_weight),
                negative_slope=self.negative_slope,
            ).squeeze(1),
            size=(t_message.shape[0], s_message.shape[0]),
            device=s_message.device,
        )
        f = torch.sparse_coo_tensor(
            indices=torch.stack([source_indices, target_indices]),
            values=F.leaky_relu(
                torch.matmul(
                    t_to_s,
                    torch.cat(
                        [
                            self.att_weight[self.target_out_channels :],
                            self.att_weight[: self.target_out_channels],
                        ]
                    ),
                ),
                negative_slope=self.negative_slope,
            ).squeeze(1),
            size=(s_message.shape[0], t_message.shape[0]),
            device=s_message.device,
        )

        if self.softmax:
            return (
                torch.sparse.softmax(e, dim=1),
                torch.sparse.softmax(f, dim=1),
            )
        return sparse_row_norm(e), sparse_row_norm(f)

    def forward(self, x_source, x_target, neighborhood):
        r"""Compute the forward pass.

        Parameters
        ----------
        x_source : torch.Tensor
            Input features of the source cells of shape
            ``[n_source_cells, source_in_channels]``.
        x_target : torch.Tensor
            Input features of the target cells of shape
            ``[n_target_cells, target_in_channels]``.
        neighborhood : torch.sparse.Tensor
            Neighborhood matrix :math:`N` of shape
            ``[n_target_cells, n_source_cells]``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Output features :math:`Y_s` (shape
            ``[n_source_cells, source_out_channels]``) and :math:`Y_t`
            (shape ``[n_target_cells, target_out_channels]``).
        """
        s_message = torch.mm(x_source, self.w_s)
        t_message = torch.mm(x_target, self.w_t)

        neighborhood_s_to_t = neighborhood.coalesce()
        neighborhood_t_to_s = neighborhood.t().coalesce()

        s_to_t_attention, t_to_s_attention = self.attention(
            s_message, t_message, neighborhood_s_to_t
        )

        neighborhood_s_to_t_att = torch.sparse_coo_tensor(
            indices=neighborhood_s_to_t.indices(),
            values=s_to_t_attention.coalesce().values()
            * neighborhood_s_to_t.values(),
            size=neighborhood_s_to_t.shape,
            device=s_message.device,
        )
        neighborhood_t_to_s_att = torch.sparse_coo_tensor(
            indices=neighborhood_t_to_s.indices(),
            values=t_to_s_attention.coalesce().values()
            * neighborhood_t_to_s.values(),
            size=neighborhood_t_to_s.shape,
            device=s_message.device,
        )

        message_on_source = torch.mm(neighborhood_t_to_s_att, t_message)
        message_on_target = torch.mm(neighborhood_s_to_t_att, s_message)

        if self.update_func is None:
            return message_on_source, message_on_target
        return self.update(message_on_source, message_on_target)


class HMCLayer(nn.Module):
    r"""Layer of the Combinatorial Complex Attention Neural Network.

    Implements the hierarchical message-passing layer of the
    combinatorial complex attention neural network of Hajij et al. 2023
    (https://arxiv.org/abs/2206.00606, Figure 35(b)), composed of two
    stacked levels of attentional message passing across the cells of
    the zeroth, first and second skeletons of a combinatorial complex.

    In the first level, 0-cells exchange messages through their
    (up) adjacency matrix and receive messages from 1-cells through the
    incidence matrix :math:`B_1`; 1-cells receive messages from 0-cells
    (via :math:`B_1^T`) and 2-cells (via :math:`B_2`); 2-cells receive
    messages from 1-cells (via :math:`B_2^T`).

    In the second level, 0-cells update through their adjacency matrix;
    1-cells update through their (up) adjacency matrix and receive
    messages from 0-cells; 2-cells update through their coadjacency
    matrix and receive messages from 1-cells.

    Parameters
    ----------
    in_channels : list of int
        Input channels for 0-, 1- and 2-cells.
    intermediate_channels : list of int
        Intermediate channels for 0-, 1- and 2-cells (output of the
        first message-passing level).
    out_channels : list of int
        Output channels for 0-, 1- and 2-cells (output of the second
        message-passing level).
    negative_slope : float, optional
        Negative slope of the LeakyReLU activation (default: 0.2).
    softmax_attention : bool, optional
        Whether to use softmax attention (default: False).
    update_func_attention : str, optional
        Activation used in the attention blocks (default: ``"relu"``).
    update_func_aggregation : str, optional
        Activation used to aggregate the messages (default: ``"relu"``).
    initialization : str, optional
        Weight initialization method (default: ``"xavier_uniform"``).
    """

    def __init__(
        self,
        in_channels,
        intermediate_channels,
        out_channels,
        negative_slope=0.2,
        softmax_attention=False,
        update_func_attention="relu",
        update_func_aggregation="relu",
        initialization="xavier_uniform",
    ):
        super().__init__()
        assert (
            len(in_channels) == 3
            and len(intermediate_channels) == 3
            and len(out_channels) == 3
        )
        in_channels_0, in_channels_1, in_channels_2 = in_channels
        int_channels_0, int_channels_1, int_channels_2 = intermediate_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels
        self.update_func_aggregation = update_func_aggregation

        common = {
            "negative_slope": negative_slope,
            "softmax": softmax_attention,
            "update_func": update_func_attention,
            "initialization": initialization,
        }

        self.hbs_0_level1 = HBS(
            source_in_channels=in_channels_0,
            source_out_channels=int_channels_0,
            **common,
        )
        self.hbns_0_1_level1 = HBNS(
            source_in_channels=in_channels_1,
            source_out_channels=int_channels_1,
            target_in_channels=in_channels_0,
            target_out_channels=int_channels_0,
            **common,
        )
        self.hbns_1_2_level1 = HBNS(
            source_in_channels=in_channels_2,
            source_out_channels=int_channels_2,
            target_in_channels=in_channels_1,
            target_out_channels=int_channels_1,
            **common,
        )
        self.hbs_0_level2 = HBS(
            source_in_channels=int_channels_0,
            source_out_channels=out_channels_0,
            **common,
        )
        self.hbns_0_1_level2 = HBNS(
            source_in_channels=int_channels_1,
            source_out_channels=out_channels_1,
            target_in_channels=int_channels_0,
            target_out_channels=out_channels_0,
            **common,
        )
        self.hbs_1_level2 = HBS(
            source_in_channels=int_channels_1,
            source_out_channels=out_channels_1,
            **common,
        )
        self.hbns_1_2_level2 = HBNS(
            source_in_channels=int_channels_2,
            source_out_channels=out_channels_2,
            target_in_channels=int_channels_1,
            target_out_channels=out_channels_1,
            **common,
        )
        self.hbs_2_level2 = HBS(
            source_in_channels=int_channels_2,
            source_out_channels=out_channels_2,
            **common,
        )

    def aggregate(self, messages):
        r"""Aggregate a list of messages by summation and activation.

        Parameters
        ----------
        messages : list of torch.Tensor
            Messages to aggregate, all of the same shape.

        Returns
        -------
        torch.Tensor
            Aggregated message.
        """
        result = messages[0]
        for message in messages[1:]:
            result = result + message
        if self.update_func_aggregation == "sigmoid":
            return torch.sigmoid(result)
        if self.update_func_aggregation == "relu":
            return F.relu(result)
        if self.update_func_aggregation == "tanh":
            return torch.tanh(result)
        return result

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_0,
        adjacency_1,
        coadjacency_2,
        incidence_1,
        incidence_2,
    ):
        r"""Compute the forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor
            Input features on 0-cells, shape ``[n_0_cells, channels]``.
        x_1 : torch.Tensor
            Input features on 1-cells, shape ``[n_1_cells, channels]``.
        x_2 : torch.Tensor
            Input features on 2-cells, shape ``[n_2_cells, channels]``.
        adjacency_0 : torch.sparse.Tensor
            Up-adjacency matrix of 0-cells,
            shape ``[n_0_cells, n_0_cells]``.
        adjacency_1 : torch.sparse.Tensor
            Up-adjacency matrix of 1-cells,
            shape ``[n_1_cells, n_1_cells]``.
        coadjacency_2 : torch.sparse.Tensor
            Coadjacency (down-adjacency) matrix of 2-cells,
            shape ``[n_2_cells, n_2_cells]``.
        incidence_1 : torch.sparse.Tensor
            Incidence matrix from 1-cells to 0-cells (:math:`B_1`),
            shape ``[n_0_cells, n_1_cells]``.
        incidence_2 : torch.sparse.Tensor
            Incidence matrix from 2-cells to 1-cells (:math:`B_2`),
            shape ``[n_1_cells, n_2_cells]``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            Updated features on the 0-, 1- and 2-cells.
        """
        # First level of message passing.
        x_0_to_0 = self.hbs_0_level1(x_0, adjacency_0)
        x_0_to_1, x_1_to_0 = self.hbns_0_1_level1(x_1, x_0, incidence_1)
        x_1_to_2, x_2_to_1 = self.hbns_1_2_level1(x_2, x_1, incidence_2)

        x_0_level1 = self.aggregate([x_0_to_0, x_1_to_0])
        x_1_level1 = self.aggregate([x_0_to_1, x_2_to_1])
        x_2_level1 = self.aggregate([x_1_to_2])

        # Second level of message passing.
        x_0_to_0 = self.hbs_0_level2(x_0_level1, adjacency_0)
        x_1_to_1 = self.hbs_1_level2(x_1_level1, adjacency_1)
        x_2_to_2 = self.hbs_2_level2(x_2_level1, coadjacency_2)

        x_0_to_1, _ = self.hbns_0_1_level2(x_1_level1, x_0_level1, incidence_1)
        x_1_to_2, _ = self.hbns_1_2_level2(x_2_level1, x_1_level1, incidence_2)

        x_0_level2 = self.aggregate([x_0_to_0])
        x_1_level2 = self.aggregate([x_0_to_1, x_1_to_1])
        x_2_level2 = self.aggregate([x_1_to_2, x_2_to_2])

        return x_0_level2, x_1_level2, x_2_level2


class HMC(nn.Module):
    r"""Hierarchical Message-passing Combinatorial Complex Attention Network.

    Stacks :class:`HMCLayer` layers implementing the combinatorial
    complex attention neural network (CCANN) of Hajij et al. 2023
    (https://arxiv.org/abs/2206.00606, Figure 35(b)), cross-checked
    against the TopoModelX reference implementation
    (``topomodelx/nn/combinatorial/hmc.py``).

    Parameters
    ----------
    in_channels : int or list of int
        Number of input channels for the 0-, 1- and 2-cells. If an int
        is given, the same number of channels is used for all ranks.
    hidden_channels : int or list of int
        Number of hidden (and output) channels for the 0-, 1- and
        2-cells. If an int is given, the same number of channels is
        used for all ranks.
    n_layers : int, optional
        Number of HMC layers (default: 2).
    negative_slope : float, optional
        Negative slope of the LeakyReLU activation (default: 0.2).
    softmax_attention : bool, optional
        Whether to use softmax attention (default: True).
    update_func_attention : str, optional
        Activation used in the attention blocks (default: ``"relu"``).
    update_func_aggregation : str, optional
        Activation used to aggregate the messages (default: ``"relu"``).
    **kwargs
        Additional arguments (ignored).
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=2,
        negative_slope=0.2,
        softmax_attention=True,
        update_func_attention="relu",
        update_func_aggregation="relu",
        **kwargs,
    ):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels] * 3
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * 3
        assert len(in_channels) == 3 and len(hidden_channels) == 3
        assert n_layers >= 1

        layer_channels = [list(in_channels)] + [
            list(hidden_channels) for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(
            [
                HMCLayer(
                    in_channels=layer_channels[i],
                    intermediate_channels=layer_channels[i + 1],
                    out_channels=layer_channels[i + 1],
                    negative_slope=negative_slope,
                    softmax_attention=softmax_attention,
                    update_func_attention=update_func_attention,
                    update_func_aggregation=update_func_aggregation,
                )
                for i in range(n_layers)
            ]
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_0,
        adjacency_1,
        coadjacency_2,
        incidence_1,
        incidence_2,
    ):
        r"""Compute the forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor
            Input features on 0-cells, shape ``[n_0_cells, channels]``.
        x_1 : torch.Tensor
            Input features on 1-cells, shape ``[n_1_cells, channels]``.
        x_2 : torch.Tensor
            Input features on 2-cells, shape ``[n_2_cells, channels]``.
        adjacency_0 : torch.sparse.Tensor
            Up-adjacency matrix of 0-cells,
            shape ``[n_0_cells, n_0_cells]``.
        adjacency_1 : torch.sparse.Tensor
            Up-adjacency matrix of 1-cells,
            shape ``[n_1_cells, n_1_cells]``.
        coadjacency_2 : torch.sparse.Tensor
            Coadjacency (down-adjacency) matrix of 2-cells,
            shape ``[n_2_cells, n_2_cells]``.
        incidence_1 : torch.sparse.Tensor
            Incidence matrix from 1-cells to 0-cells (:math:`B_1`),
            shape ``[n_0_cells, n_1_cells]``.
        incidence_2 : torch.sparse.Tensor
            Incidence matrix from 2-cells to 1-cells (:math:`B_2`),
            shape ``[n_1_cells, n_2_cells]``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            Final features on the 0-, 1- and 2-cells.
        """
        for layer in self.layers:
            x_0, x_1, x_2 = layer(
                x_0,
                x_1,
                x_2,
                adjacency_0,
                adjacency_1,
                coadjacency_2,
                incidence_1,
                incidence_2,
            )
        return x_0, x_1, x_2
