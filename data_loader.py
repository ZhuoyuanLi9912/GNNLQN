import torch
from scipy.io import loadmat
from torch_geometric.data import HeteroData


def load_lqn_dataset(file_path, device='cuda'):
    """
    Loads the LQN dataset from a .mat file, converts all graphs to PyTorch Geometric format,
    and normalizes the data.
    """
    mat_data = loadmat(file_path)
    LQN_dataset = mat_data['LQN_dataset']

    # Iterate through all cells in the 20000x1 dataset
    graphs = []
    for i in range(LQN_dataset.shape[0]):  # Iterate over all 20,000 rows
        graph = convert_lqn_to_hetero_data(LQN_dataset[i][0], device)
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




def normalize_attributes(data, feature_range=(0, 1)):
    """
    Applies Min-Max normalization to each attribute (column) of the data independently.

    Args:
        data (torch.Tensor): A tensor of shape [num_samples, num_attributes].
        feature_range (tuple): Desired range of transformed data (default is (0, 1)).

    Returns:
        torch.Tensor: Min-Max normalized tensor with the same shape as input.
    """
    min_val = data.min(dim=0, keepdim=True).values
    max_val = data.max(dim=0, keepdim=True).values
    scale = feature_range[1] - feature_range[0]
    min_range = feature_range[0]
    # Avoid division by zero for constant features
    denom = max_val - min_val
    denom[denom == 0] = 1.0  # Avoid division by zero
    return (data - min_val) / denom * scale + min_range


def convert_lqn_to_hetero_data(LQN_graph, device='cuda'):
    """
    Converts a single LQN graph structure to a PyTorch Geometric HeteroData object.
    """
    data = HeteroData()

    # Extract field names from the structured array
    fields = LQN_graph.dtype.names

    # Normalize node attributes
    if 'processor_attributes' in fields:
        data['processor'].x = normalize_attributes(
            torch.tensor(LQN_graph['processor_attributes'][0][0], dtype=torch.float, device=device))
    if 'task_attributes' in fields:
        data['task'].x = normalize_attributes(
            torch.tensor(LQN_graph['task_attributes'][0][0], dtype=torch.float, device=device))
    if 'entry_attributes' in fields:
        data['entry'].x = normalize_attributes(
            torch.tensor(LQN_graph['entry_attributes'][0][0], dtype=torch.float, device=device))

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
        data['entry', 'calls', 'entry'].edge_attr = normalize_attributes(
            torch.tensor(LQN_graph['entry_call_entry_edge_attributes'][0][0], dtype=torch.float, device=device))

    # Labels
    if 'entry_queue_lengths' in fields:
        data['entry'].y = normalize_attributes(
            torch.tensor(LQN_graph['entry_queue_lengths'][0][0], dtype=torch.float, device=device))

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
    if 'entry_queue_lengths' in fields:
        data['entry'].y = torch.tensor(LQN_graph['entry_queue_lengths'][0][0], dtype=torch.float, device=device)

    return data

