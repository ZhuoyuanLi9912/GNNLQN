import torch
from torch_geometric.loader import DataLoader
from data_loader import load_lqn_dataset


def verify_hetero_data(data):
    """
    Verifies and prints details about a HeteroData object, including node types,
    edge types, attributes, connections, and labels.
    """
    print("HeteroData Summary:")
    print(data)

    print("\nNode Types:", data.node_types)
    for node_type in data.node_types:
        print(f"\n{node_type.capitalize()} Node Features (x):")
        print(data[node_type].x)
        print(f"Shape: {data[node_type].x.shape}")

    print("\nEdge Types:", data.edge_types)
    for edge_type in data.edge_types:
        print(f"\n{edge_type} Edge Indices:")
        print(data[edge_type].edge_index)
        print(f"Shape: {data[edge_type].edge_index.shape}")
        if 'edge_attr' in data[edge_type]:
            print(f"Edge Attributes for {edge_type}:")
            print(data[edge_type].edge_attr)
            print(f"Shape: {data[edge_type].edge_attr.shape}")

    if 'y' in data['entry']:
        print("\nLabels for 'Entry' Nodes (y):")
        print(data['entry'].y)
        print(f"Shape: {data['entry'].y.shape}")


def main():
    # Path to your .mat file
    mat_file_path = r"C:\PhD\line_2024\LQN_GNN_data_collection\test.mat"

    # Set the device to CUDA
    device = torch.device('cuda')

    # Load the dataset directly on the CUDA device
    print("Loading dataset onto GPU...")
    graphs = load_lqn_dataset(mat_file_path, device=device)

    # Verify the number of graphs loaded
    print(f"\nNumber of graphs loaded: {len(graphs)}\n")

    # Inspect the first graph
    first_graph = graphs[0]
    print("\nInspecting the First Graph:")
    verify_hetero_data(first_graph)

    # Test with a DataLoader
    print("\nTesting with PyTorch Geometric DataLoader...")
    batch_size = 4
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    # Iterate through one batch
    for batch in loader:
        batch = batch.to(device)  # Ensure the batch is on CUDA
        print("\nBatch loaded (on CUDA):")
        print(batch)
        print("Batch size:", batch.num_graphs)  # Should match the batch size
        print("Batch device for 'entry' nodes:", batch['entry'].x.device)
        break


if __name__ == '__main__':
    main()
