import torch
from scipy.io import loadmat
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler


def load_lqn_dataset(file_path, device='cuda'):
    """
    Loads the LQN dataset from a .mat file, converts all graphs to PyTorch Geometric format,
    and applies global normalization to node and edge attributes, including labels.
    """
    mat_data = loadmat(file_path)
    LQN_dataset = mat_data['LQN_dataset']

    # Initialize lists to collect all attributes globally
    processor_attributes = []
    task_attributes = []
    entry_attributes = []
    edge_attributes = []
    labels = []  # Collect labels (entry_throughputs)

    # Step 1: Collect attributes and labels from all graphs
    for i in range(LQN_dataset.shape[0]):
        LQN_graph = LQN_dataset[i][0]
        fields = LQN_graph.dtype.names

        # Collect processor node attributes
        if 'processor_attributes' in fields:
            processor_attributes.append(
                torch.tensor(LQN_graph['processor_attributes'][0][0], dtype=torch.float)
            )
        # Collect task node attributes
        if 'task_attributes' in fields:
            task_attributes.append(
                torch.tensor(LQN_graph['task_attributes'][0][0], dtype=torch.float)
            )
        # Collect entry node attributes
        if 'entry_attributes' in fields:
            entry_attributes.append(
                torch.tensor(LQN_graph['entry_attributes'][0][0], dtype=torch.float)
            )
        # Collect edge attributes
        if 'entry_call_entry_edge_attributes' in fields:
            edge_attributes.append(
                torch.tensor(LQN_graph['entry_call_entry_edge_attributes'][0][0], dtype=torch.float)
            )
        # Collect labels (entry queue lengths)
        if 'entry_throughputs' in fields:
            labels.append(
                torch.tensor(LQN_graph['entry_throughputs'][0][0], dtype=torch.float)
            )

    # Concatenate all attributes and labels for global normalization
    processor_attributes = torch.cat(processor_attributes, dim=0)
    task_attributes = torch.cat(task_attributes, dim=0)
    entry_attributes = torch.cat(entry_attributes, dim=0)
    edge_attributes = torch.cat(edge_attributes, dim=0)
    labels = torch.cat(labels, dim=0)

    # Apply Min-Max normalization globally
    processor_scaler = MinMaxScaler()
    task_scaler = MinMaxScaler()
    entry_scaler = MinMaxScaler()
    edge_scaler = MinMaxScaler()
    label_scaler = MinMaxScaler()

    normalized_processor_attributes = torch.tensor(
        processor_scaler.fit_transform(processor_attributes.numpy()), dtype=torch.float, device=device
    )
    normalized_task_attributes = torch.tensor(
        task_scaler.fit_transform(task_attributes.numpy()), dtype=torch.float, device=device
    )
    normalized_entry_attributes = torch.tensor(
        entry_scaler.fit_transform(entry_attributes.numpy()), dtype=torch.float, device=device
    )
    normalized_edge_attributes = torch.tensor(
        edge_scaler.fit_transform(edge_attributes.numpy()), dtype=torch.float, device=device
    )
    normalized_labels = torch.tensor(
        label_scaler.fit_transform(labels.numpy()), dtype=torch.float, device=device
    )

    # Step 2: Reassign normalized attributes and labels back to graphs
    normalized_processor_index = 0
    normalized_task_index = 0
    normalized_entry_index = 0
    normalized_edge_index = 0
    normalized_label_index = 0

    graphs = []
    for i in range(LQN_dataset.shape[0]):
        LQN_graph = LQN_dataset[i][0]
        graph = convert_lqn_to_hetero_data(
            LQN_graph, device,
            normalized_processor_attributes[normalized_processor_index:normalized_processor_index +
                                           LQN_graph['processor_attributes'][0][0].shape[0]],
            normalized_task_attributes[normalized_task_index:normalized_task_index +
                                       LQN_graph['task_attributes'][0][0].shape[0]],
            normalized_entry_attributes[normalized_entry_index:normalized_entry_index +
                                        LQN_graph['entry_attributes'][0][0].shape[0]],
            normalized_edge_attributes[normalized_edge_index:normalized_edge_index +
                                       LQN_graph['entry_call_entry_edge_attributes'][0][0].shape[0]],
            normalized_labels[normalized_label_index:normalized_label_index +
                              LQN_graph['entry_throughputs'][0][0].shape[0]]
        )
        normalized_processor_index += LQN_graph['processor_attributes'][0][0].shape[0]
        normalized_task_index += LQN_graph['task_attributes'][0][0].shape[0]
        normalized_entry_index += LQN_graph['entry_attributes'][0][0].shape[0]
        normalized_edge_index += LQN_graph['entry_call_entry_edge_attributes'][0][0].shape[0]
        normalized_label_index += LQN_graph['entry_throughputs'][0][0].shape[0]

        graphs.append(graph)

    return graphs


def adjust_edge_index(edge_index):
    """
    Adjusts edge indices from MATLAB 1-based indexing to Python 0-based indexing.

    Args:
        edge_index (torch.Tensor): Edge indices.

    Returns:
        torch.Tensor: Adjusted edge indices.
    """
    return edge_index - 1


def convert_lqn_to_hetero_data(LQN_graph, device, processor_norm, task_norm, entry_norm, edge_norm, label_norm):
    """
    Converts a single LQN graph structure to a PyTorch Geometric HeteroData object with globally normalized attributes and labels.
    """
    data = HeteroData()

    # Extract field names from the structured array
    fields = LQN_graph.dtype.names

    # Assign normalized node attributes
    if 'processor_attributes' in fields:
        data['processor'].x = processor_norm
    if 'task_attributes' in fields:
        data['task'].x = task_norm
    if 'entry_attributes' in fields:
        data['entry'].x = entry_norm

    # Assign edge indices (adjusted with a function)
    if 'task_on_processor_edges' in fields:
        data['task', 'on', 'processor'].edge_index = adjust_edge_index(
            torch.tensor(LQN_graph['task_on_processor_edges'][0][0], dtype=torch.long, device=device))
    if 'entry_on_task_edges' in fields:
        data['entry', 'on', 'task'].edge_index = adjust_edge_index(
            torch.tensor(LQN_graph['entry_on_task_edges'][0][0], dtype=torch.long, device=device))
    if 'entry_call_entry_edges' in fields:
        data['entry', 'calls', 'entry'].edge_index = adjust_edge_index(
            torch.tensor(LQN_graph['entry_call_entry_edges'][0][0], dtype=torch.long, device=device))

    # Assign normalized edge attributes
    if 'entry_call_entry_edge_attributes' in fields:
        data['entry', 'calls', 'entry'].edge_attr = edge_norm

    # Assign normalized labels
    if 'entry_throughputs' in fields:
        data['entry'].y = label_norm

    return data


def convert_lqn_to_hetero_data_test(LQN_graph, device='cuda'):
    """
    Converts a single LQN graph structure to a PyTorch Geometric HeteroData object.
    """
    data = HeteroData()

    # Extract field names from the structured array
    fields = LQN_graph.dtype.names

    # Normalize node attributes
    if 'processor_attributes' in fields:
        data['processor'].x = torch.tensor(LQN_graph['processor_attributes'][0][0], dtype=torch.float, device=device)
    if 'task_attributes' in fields:
        data['task'].x = torch.tensor(LQN_graph['task_attributes'][0][0], dtype=torch.float, device=device)
    if 'entry_attributes' in fields:
        data['entry'].x = torch.tensor(LQN_graph['entry_attributes'][0][0], dtype=torch.float, device=device)

    # Edge indices (adjusted with a function)
    if 'task_on_processor_edges' in fields:
        data['task', 'on', 'processor'].edge_index = adjust_edge_index(
            torch.tensor(LQN_graph['task_on_processor_edges'][0][0], dtype=torch.long, device=device))
    if 'entry_on_task_edges' in fields:
        data['entry', 'on', 'task'].edge_index = adjust_edge_index(
            torch.tensor(LQN_graph['entry_on_task_edges'][0][0], dtype=torch.long, device=device))
    if 'entry_call_entry_edges' in fields:
        data['entry', 'calls', 'entry'].edge_index = adjust_edge_index(
            torch.tensor(LQN_graph['entry_call_entry_edges'][0][0], dtype=torch.long, device=device))

    # Edge attributes
    if 'entry_call_entry_edge_attributes' in fields:
        data['entry', 'calls', 'entry'].edge_attr = torch.tensor(LQN_graph['entry_call_entry_edge_attributes'][0][0], dtype=torch.float, device=device)

    # Labels
    if 'entry_throughputs' in fields:
        data['entry'].y = torch.tensor(LQN_graph['entry_throughputs'][0][0], dtype=torch.float, device=device)

    return data

