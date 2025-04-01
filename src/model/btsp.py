import dataclasses

import torch

@dataclasses.dataclass
class BinaryBTSPLayerParams:
    """Parameter Dataclass for BTSP layer with 0-1 binary weights."""

    input_dim: int
    memory_neurons: int
    fq: float
    fw: float
    device: str
    fast_btsp: bool = False
    # dtype: torch.dtype = torch.float # not available for now
    # due to the nature of the layer, the dtype is bool


class BinaryBTSPLayer():
    """This is the class for BTSP layer with 0-1 binary weights.
    
    TODO(Ruhao Tian): make this layer type independent.

    Attributes:
        input_dim (int): The input dimension.
        memory_neurons (int): The number of memory neurons.
        fq: plateau potential possibility
        fw: connection ratio between neurons
        device: The device to deploy the layer.
        weights: The weights of the layer.
        connection_matrix: The matrix describing which neurons are connected.
    """

    def __init__(self, params: BinaryBTSPLayerParams) -> None:
        """Initialize the layer."""
        self.input_dim = params.input_dim
        self.memory_neurons = params.memory_neurons
        self.fq = params.fq
        self.fw = params.fw
        self.device = params.device
        self.fast_btsp = params.fast_btsp
        self.dtype = torch.bool
        self.weights = None
        self.connection_matrix = None
        self.weight_reset()

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        output_data = torch.matmul(input_data, self.weights.to(input_data.dtype))
        return output_data

    def learn_and_forward(self, training_data: torch.Tensor) -> torch.Tensor:
        """One-shot learning while forward pass.

        Args:
            training_data (torch.Tensor): The training data, the same as normal
                 input data.
        """

        fq_half = self.fq / 2

        with torch.no_grad():
            # plateau weight change possibility document if each neuron has weight
            # change possibiltiy when receiving a memory item
            # shape: (batch_size, memory_neurons)
            plateau_weight_change_possibility = (
                torch.rand(
                    training_data.shape[0], self.memory_neurons, device=self.device
                )
                < fq_half
            )

            # plateau weight change synapse document if each synapse has plateau
            # potential when receiving memory item
            plateau_weight_change_synapse = plateau_weight_change_possibility.unsqueeze(
                1
            )

            # weight change allowance synapse document if each synapse
            # satisfies the plateau potential condition and the connection matrix
            # shape: (batch_size, input_dim, memory_neurons)
            weight_change_allowance_synapse = (
                plateau_weight_change_synapse * self.connection_matrix
            )

            # weight_change_sequence is a binary matrix, indicating the update of
            # each weight during the training process
            weight_change_sequence = (
                weight_change_allowance_synapse * training_data.unsqueeze(2)
            )
            
            if self.fast_btsp:
                # cumsum is a very computationally expensive operation
                # btsp can also be approximated by using the last weight
                weight_change_sum = weight_change_sequence.sum(dim=0) % 2
                
                self.weights = torch.where(
                    weight_change_sum > 0, ~self.weights, self.weights
                )
                
                output_data = torch.matmul(training_data.float(), self.weights.float())
            
            else:
                # weight_change_sum is the number of total weight changes for each synapse
                # as weights are binary, the sum is the number of changes
                # shape: (batch_size, input_dim, memory_neurons)
                weight_change_sum = torch.cumsum(weight_change_sequence.int(), dim=0) % 2

                # weight sequence is the weight after each training data
                weight_sequence = torch.where(
                    weight_change_sum > 0, ~self.weights, self.weights
                )

                # update the weights
                # final weight is stored in the last element of the weight_sequence
                self.weights = weight_sequence[-1]

                # calculate output DURING learning
                # shape: (batch_size, memory_neurons)
                output_data = torch.bmm(
                    training_data.unsqueeze(1).float(), weight_sequence.float()
                )

                # remove the neuron dimension
                output_data = output_data.squeeze(1)
            return output_data

    def learn(self, training_data: torch.Tensor) -> None:
        """This is basically the same as learn_and_forward-.

        TODO(Ruhao Tian): refactor this to avoid code duplication.
        """
        self.learn_and_forward(training_data)

    def weight_reset(self, *args, **kwargs) -> None:
        """Reset the weights."""
        self.weights = (
            torch.rand((self.input_dim, self.memory_neurons), device=self.device)
            < self.fq
        ).to(self.dtype)
        if "weight" in kwargs:
            if kwargs["weight"] is not None:
                self.weights = kwargs["weight"]
        self.connection_matrix = (
            torch.rand((self.input_dim, self.memory_neurons), device=self.device)
            < self.fw
        ).to(self.dtype)
        return

@dataclasses.dataclass
class ContinuousBTSPLayerParams:
    """Parameter Dataclass for BTSP layer with continuous weights."""

    input_dim: int
    memory_neurons: int
    fq: float
    fw: float
    device: str
    theta: float = 0.6 # threshold for weight change
    
class ContinuousBTSPLayer():
    """This is the class for BTSP layer with continuous weights.
    """
    
    def __init__(self, params: ContinuousBTSPLayerParams):
        """Initialize the layer.

        Args:
            params (ContinuousBTSPLayerParams): The parameters for the layer.
        """
        self.input_dim = params.input_dim
        self.memory_neurons = params.memory_neurons
        self.fq = params.fq
        self.fw = params.fw
        self.theta = params.theta
        self.device = params.device
        self.weights = None
        self.connection_matrix = None
        self.weight_reset()
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        output_data = torch.matmul(input_data, self.weights.to(input_data.dtype))
        return output_data
    
    def learn_single_pattern(self, training_data: torch.Tensor) -> torch.Tensor:
        """Learn a single pattern."""
        
        # check the input shape: (1, input_dim)
        assert training_data.shape[0] == 1
        assert training_data.shape[1] == self.input_dim
        
        # create plateau potential matrix
        plateau_potential = (torch.rand((1, self.memory_neurons), device=self.device) < self.fq).float()
        # expand the plateau potential matrix to the shape of the weights
        plateau_potential = plateau_potential.expand(self.input_dim, self.memory_neurons)
        
        # expand the training data to the shape of the weights
        expanded_input = training_data.transpose(0, 1).expand(self.input_dim, self.memory_neurons).float()
        
        # decay factor: (1 - weights) * plateau_potential * input_data
        decay_factor = (torch.ones_like(self.weights) - self.weights) * plateau_potential * expanded_input
        
        # change allowance: abs(plateau_potential * input_data) > theta
        change_allowance = (torch.abs(plateau_potential * expanded_input) > self.theta).float()
        
        # change factor: - change_allowance * weights
        change_factor = - change_allowance * self.weights
        
        # total weight change: decay_factor + change_factor
        total_weight_change = decay_factor + change_factor
        
        # apply weight change
        self.weights = self.weights + total_weight_change
        
        # apply connection matrix
        self.weights = self.weights * self.connection_matrix
        
        return self.forward(training_data)
    
    def weight_reset(self) -> None:
        
        # reset the connection matrix
        self.connection_matrix = (
            torch.rand((self.input_dim, self.memory_neurons), device=self.device)
            < self.fw
        ).float()
        
        # reset the weights
        # use lognormal distribution to initialize the weights
        # Sen Song(2005) Highly Nonrandom Features of Synaptic Connectivity in Local Cortical Circuits
        #self.weights = torch.distributions.log_normal.LogNormal(0.5, 0.5).sample((self.input_dim, self.memory_neurons)).to(self.device)
        
        self.weights = torch.zeros((self.input_dim, self.memory_neurons), device=self.device)
        
        # apply connection matrix to the weights
        self.weights = self.weights * self.connection_matrix