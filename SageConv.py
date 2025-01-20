import torch
from torch.nn import Linear, LeakyReLU, ModuleList
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from data_loader import load_lqn_dataset
from torch_scatter import scatter_add


class LQNGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1, edge_attr_dim=4, num_iterations=2):
        super(LQNGNN, self).__init__()
        self.num_iterations = num_iterations

        # Input encoders for node features
        self.entry_encoder = Linear(2, hidden_channels)  # Entry has 2 features
        self.task_encoder = Linear(3, hidden_channels)   # Task has 3 features
        self.processor_encoder = Linear(1, hidden_channels)  # Processor has 1 feature

        # Edge encoder for entry_call_entry edges
        self.edge_encoder = Linear(edge_attr_dim, hidden_channels)  # Edge attr has 4 features, outputs hidden_channels

        # Aggregation layers for bidirectional message passing
        self.entry_to_task_layers = ModuleList(
            [SAGEConv((hidden_channels, hidden_channels), hidden_channels) for _ in range(num_iterations)]
        )
        self.task_to_entry_layers = ModuleList(
            [SAGEConv((hidden_channels, hidden_channels), hidden_channels) for _ in range(num_iterations)]
        )
        self.task_to_processor_layers = ModuleList(
            [SAGEConv((hidden_channels, hidden_channels), hidden_channels) for _ in range(num_iterations)]
        )
        self.processor_to_task_layers = ModuleList(
            [SAGEConv((hidden_channels, hidden_channels), hidden_channels) for _ in range(num_iterations)]
        )
        self.entry_to_entry_layers = ModuleList(
            [SAGEConv((hidden_channels, hidden_channels), hidden_channels) for _ in range(num_iterations)]
        )
        self.entry_to_entry_reverse_layers = ModuleList(
            [SAGEConv((hidden_channels, hidden_channels), hidden_channels) for _ in range(num_iterations)]
        )

        # Final aggregation layers for entry nodes
        self.entry_final_from_task = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.entry_final_from_entry = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.entry_final_from_entry_reverse = SAGEConv((hidden_channels, hidden_channels), hidden_channels)

        # Output layer for predicting entry queue lengths
        self.entry_predictor = Linear(hidden_channels, out_channels)

        # Activation function
        self.activation = LeakyReLU(negative_slope=0.1)

    def forward(self, data):
        # Encode raw node features with activation
        entry_x = self.activation(self.entry_encoder(data['entry'].x))
        task_x = self.activation(self.task_encoder(data['task'].x))
        processor_x = self.activation(self.processor_encoder(data['processor'].x))

        # Encode edge attributes for entry_call_entry edges
        if 'edge_attr' in data['entry', 'calls', 'entry']:
            edge_embeddings = self.activation(self.edge_encoder(data['entry', 'calls', 'entry'].edge_attr))
        else:
            edge_embeddings = None

        # Iterative message passing
        for i in range(self.num_iterations):
            # Entry node aggregation
            if edge_embeddings is not None:
                # Aggregate edge embeddings into the source nodes
                edge_index = data['entry', 'calls', 'entry'].edge_index
                aggregated_edge_embeddings = scatter_add(edge_embeddings, edge_index[0], dim=0, dim_size=entry_x.size(0))
                entry_x = entry_x + aggregated_edge_embeddings

            task_to_entry = self.task_to_entry_layers[i](
                (task_x, entry_x), data['entry', 'on', 'task'].edge_index[[1, 0]]
            )
            entry_to_entry = self.entry_to_entry_layers[i](
                (entry_x, entry_x), data['entry', 'calls', 'entry'].edge_index
            )
            entry_to_entry_reverse = self.entry_to_entry_reverse_layers[i](
                (entry_x, entry_x), data['entry', 'calls', 'entry'].edge_index[[1, 0]]
            )
            entry_x = self.activation(task_to_entry + entry_to_entry + entry_to_entry_reverse)

            # Task node aggregation
            entry_to_task = self.entry_to_task_layers[i](
                (entry_x, task_x), data['entry', 'on', 'task'].edge_index
            )
            processor_to_task = self.processor_to_task_layers[i](
                (processor_x, task_x), data['task', 'on', 'processor'].edge_index[[1, 0]]
            )
            task_x = self.activation(entry_to_task + processor_to_task)

            # Processor node aggregation
            task_to_processor = self.task_to_processor_layers[i](
                (task_x, processor_x), data['task', 'on', 'processor'].edge_index
            )
            processor_x = self.activation(task_to_processor)

        # Final round for entry nodes
        if edge_embeddings is not None:
            aggregated_edge_embeddings = scatter_add(edge_embeddings, edge_index[0], dim=0, dim_size=entry_x.size(0))
            entry_x = entry_x + aggregated_edge_embeddings

        entry_final_from_task = self.entry_final_from_task(
            (task_x, entry_x), data['entry', 'on', 'task'].edge_index[[1, 0]]
        )
        entry_final_from_entry = self.entry_final_from_entry(
            (entry_x, entry_x), data['entry', 'calls', 'entry'].edge_index
        )
        entry_final_from_entry_reverse = self.entry_final_from_entry_reverse(
            (entry_x, entry_x), data['entry', 'calls', 'entry'].edge_index[[1, 0]]
        )
        entry_x = self.activation(entry_final_from_task + entry_final_from_entry +entry_final_from_entry_reverse)

        # Predict queue lengths for entry nodes
        queue_lengths = self.entry_predictor(entry_x)

        # Ensure positive predictions
        queue_lengths = self.activation(queue_lengths)

        return queue_lengths


def mape_loss(predictions, targets, epsilon=1e-8):
    """
    Computes the Mean Absolute Percentage Error (MAPE) loss.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: MAPE loss.
    """
    return torch.mean(torch.abs(targets - predictions) / (torch.abs(targets) + epsilon))


def compute_average_relative_error(predictions, labels):
    """Compute the average absolute relative error."""
    return torch.mean(torch.abs(predictions - labels) / (labels + 1e-6)).item()


def train_model(file_path, num_epochs=50, hidden_channels=32, edge_attr_dim=4, learning_rate=0.001, device='cuda'):
    # Load the dataset using the custom loader
    graphs = load_lqn_dataset(file_path, device=device)

    # Split the dataset into 8:1:1 (train, validation, test)
    train_size = int(0.8 * len(graphs))
    val_size = int(0.1 * len(graphs))
    test_size = len(graphs) - train_size - val_size
    train_data, val_data, test_data = random_split(graphs, [train_size, val_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize the model
    model = LQNGNN(hidden_channels, out_channels=1, edge_attr_dim=edge_attr_dim, num_iterations=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = mape_loss  # Using MAPE as the loss function

    best_val_loss = float('inf')  # Keep track of the best validation loss
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_are = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch)
            loss = criterion(predictions, batch['entry'].y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_are += compute_average_relative_error(predictions, batch['entry'].y)
        train_loss /= len(train_loader)
        train_are /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_are = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                predictions = model(batch)
                val_loss += criterion(predictions, batch['entry'].y).item()
                val_are += compute_average_relative_error(predictions, batch['entry'].y)
        val_loss /= len(val_loader)
        val_are /= len(val_loader)

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Log the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss (MAPE): {train_loss:.4f}, Train ARE: {train_are:.4f}, "
              f"Val Loss (MAPE): {val_loss:.4f}, Val ARE: {val_are:.4f}, Learning Rate: {current_lr:.6f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    print("Training completed!")
    return model, test_loader


def test_model(model, test_loader, device='cuda'):
    model.eval()
    test_loss = 0
    test_are = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            predictions = model(batch)
            test_loss += mape_loss(predictions, batch['entry'].y).item()
            test_are += compute_average_relative_error(predictions, batch['entry'].y)
    test_loss /= len(test_loader)
    test_are /= len(test_loader)

    print(f"Test Loss (MAPE): {test_loss:.4f}, Test ARE: {test_are:.4f}")





def main():
    # File path to your dataset
    file_path = r"C:\PhD\line_2024\test_overall.mat" # Replace with the actual path to your .mat dataset

    # Hyperparameters
    num_epochs = 50
    hidden_channels = 64
    edge_attr_dim = 4

    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Train the model
    print("Starting training...")
    model, test_loader = train_model(
        file_path=file_path,
        num_epochs=num_epochs,
        hidden_channels=hidden_channels,
        edge_attr_dim=edge_attr_dim,

        learning_rate=learning_rate,
        device=device
    )
    print("Training completed!")

    # Test the model
    print("Starting testing...")
    test_model(model, test_loader, device=device)
    print("Testing completed!")

if __name__ == "__main__":
    main()
