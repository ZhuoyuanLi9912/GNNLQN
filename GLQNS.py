import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# Entry Message-Passing Class
class EntryMessagePassing(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')  # Aggregation: sum
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)  # GRU for entry nodes
        self.prob_weight = nn.Linear(hidden_dim * 3, hidden_dim)  # Edge-specific weight
        self.task_weight = nn.Linear(hidden_dim * 2, hidden_dim)  # Task-entry relationship weight

    def forward(self, x, edge_index, edge_attr, task_messages):
        # Extract edge attribute `rho` (probability of sending a call)
        rho = edge_attr[:, 0]

        # Message-passing for entry nodes
        messages = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, rho=rho)

        # Combine task messages and propagate through GRU
        combined_messages = messages + self.task_weight(task_messages)
        updated_x = self.gru(x, combined_messages)
        return updated_x

    def message(self, x_i, x_j, edge_attr, rho):
        # Calculate messages for entry nodes
        concatenated = torch.cat([x_i, x_j, edge_attr], dim=-1)  # Concatenate x_i, x_j, and edge_attr
        return rho.view(-1, 1) * self.prob_weight(concatenated)  # Weighted by rho


# Task Message-Passing Class
class TaskMessagePassing(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')  # Aggregation: sum
        self.hidden_dim = hidden_dim
        self.entry_weight = nn.Linear(hidden_dim, hidden_dim)  # Entry-task weight
        self.processor_weight = nn.Linear(hidden_dim, hidden_dim)  # Processor-task weight
        self.self_weight = nn.Linear(hidden_dim, hidden_dim)  # Task self-embedding weight

    def forward(self, x, edge_index, processor_message, num_entries):
        # Propagate messages from neighboring entry nodes
        messages = self.propagate(edge_index=edge_index, x=x)

        # Divide by the number of entries per task
        messages = messages / num_entries.view(-1, 1)

        # Combine with processor messages and self-node information
        combined_messages = F.leaky_relu(
            messages + self.processor_weight(processor_message) + self.self_weight(x)
        )
        return combined_messages

    def message(self, x_j):
        # Calculate messages from neighboring entry nodes
        return self.entry_weight(x_j)


# Processor Message-Passing Class
class ProcessorMessagePassing(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')  # Aggregation: sum
        self.hidden_dim = hidden_dim
        self.task_weight = nn.Linear(hidden_dim, hidden_dim)  # Task-processor weight
        self.self_weight = nn.Linear(hidden_dim, hidden_dim)  # Processor self-embedding weight

    def forward(self, x, edge_index, num_tasks):
        # Propagate messages from neighboring task nodes
        messages = self.propagate(edge_index=edge_index, x=x)

        # Divide by the number of tasks per processor
        messages = messages / num_tasks.view(-1, 1)

        # Combine with self-node information
        combined_messages = F.leaky_relu(messages + self.self_weight(x))
        return combined_messages

    def message(self, x_j):
        # Calculate messages from neighboring task nodes
        return self.task_weight(x_j)


# Custom GNN Model
class LQNGNN(torch.nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, num_iterations=2):
        """
        Custom message-passing GNN for LQN graphs.

        Args:
            input_dims (dict): Input dimensions for each node type.
                               Example: {'processor': 1, 'task': 3, 'entry': 2}.
            hidden_dim (int): Hidden dimension for embeddings.
            output_dim (int): Output dimension for predictions.
            num_iterations (int): Number of message-passing rounds.
        """
        super().__init__()
        self.num_iterations = num_iterations

        # Initial embedding layers for each node type
        self.lin_dict = nn.ModuleDict({
            node_type: nn.Linear(input_dim, hidden_dim)
            for node_type, input_dim in input_dims.items()
        })

        # Message-passing layers for each node type
        self.entry_mp = EntryMessagePassing(hidden_dim)
        self.task_mp = TaskMessagePassing(hidden_dim)
        self.processor_mp = ProcessorMessagePassing(hidden_dim)

        # MLP for point-wise regression on entry nodes
        self.entry_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        """
        Forward pass for the custom GNN.

        Args:
            data (HeteroData): Heterogeneous graph data.

        Returns:
            torch.Tensor: Predictions for 'entry' nodes.
        """
        # Initial embedding for each node type
        x_dict = {node_type: self.lin_dict[node_type](data[node_type].x) for node_type in data.node_types}

        # Extract edge indices and attributes
        entry_edge_index = data['entry', 'calls', 'entry'].edge_index
        entry_edge_attr = data['entry', 'calls', 'entry'].edge_attr
        task_edge_index = data['entry', 'on', 'task'].edge_index
        processor_edge_index = data['task', 'on', 'processor'].edge_index

        # Dynamically count num_entries (entries per task)
        num_entries = torch.bincount(task_edge_index[1], minlength=data['task'].x.size(0)).float()

        # Dynamically count num_tasks (tasks per processor)
        num_tasks = torch.bincount(processor_edge_index[1], minlength=data['processor'].x.size(0)).float()

        # Iterative message-passing
        for _ in range(self.num_iterations):
            # Update entry nodes
            x_dict['entry'] = self.entry_mp(
                x_dict['entry'], entry_edge_index, entry_edge_attr, x_dict['task']
            )

            # Update task nodes
            x_dict['task'] = self.task_mp(
                x_dict['task'], task_edge_index, x_dict['processor'], num_entries
            )

            # Update processor nodes
            x_dict['processor'] = self.processor_mp(
                x_dict['processor'], processor_edge_index, num_tasks
            )

        # Final step: Back to entry nodes
        x_dict['entry'] = self.entry_mp(
            x_dict['entry'], entry_edge_index, entry_edge_attr, x_dict['task']
        )

        # Final MLP-based predictions for entry nodes
        out = self.entry_mlp(x_dict['entry'])
        return out
