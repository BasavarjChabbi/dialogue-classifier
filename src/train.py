import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DialogueDataset, collate_fn

# 1. Define the Bidirectional LSTM Model Architecture
class DialogueBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(DialogueBiLSTM, self).__init__()
        # Embedding layer turns token IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Bidirectional LSTM processes sentences forwards and backwards
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Fully connected layer maps hidden states to our classes (hidden_dim * 2 because bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        # hidden shape: [num_layers * num_directions, batch_size, hidden_dim]
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        return self.fc(out)

# 2. Main Training Execution Loop
def main():
    # Load dataset and prepare data loader
    db_path = 'data/dialogue_db.sqlite'
    dataset = DialogueDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Hyperparameters
    vocab_size = len(dataset.vocab)
    num_classes = len(dataset.label_map)
    embed_dim = 16
    hidden_dim = 32
    epochs = 10
    learning_rate = 0.01

    # Initialize model, loss function, and optimizer
    model = DialogueBiLSTM(vocab_size, embed_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(" Starting Training Loop...\n")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_texts, batch_labels in dataloader:
            # Clear historical gradients
            optimizer.zero_grad()
            
            # Forward pass: compute predictions
            predictions = model(batch_texts)
            
            # Compute cross entropy loss
            loss = criterion(predictions, batch_labels)
            
            # Backward pass: compute updates
            loss.backward()
            
            # Step optimizer: adjust weights
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Total Loss: {epoch_loss:.4f}")

    print("\nTraining Complete! Model optimized successfully.")

if __name__ == "__main__":
    main()