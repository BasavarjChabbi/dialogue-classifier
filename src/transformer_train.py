import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # <-- Import AdamW natively from PyTorch instead!
from transformers import DistilBertForSequenceClassification
from transformer_dataset import TransformerDialogueDataset

def main():
    db_path = 'data/dialogue_db.sqlite'
    dataset = TransformerDialogueDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize Pre-trained Transformer with our specific target class count
    num_classes = len(dataset.label_map)
   # Initialize Pre-trained Transformer with our specific target class count
    num_classes = len(dataset.label_map)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=num_classes  # <-- Changed parameter name here!
    )
    # Use AdamW - the optimized weight decay version of Adam built for Transformers
    optimizer = AdamW(model.parameters(), lr=5e-5)

    print(" Starting Transformer Fine-Tuning Loop...\n")
    model.train()
    
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Feed inputs and attention masks directly to BERT
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['label']
            )
            
            # Hugging Face models calculate loss automatically when given a label parameter
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Fine-Tuning Loss: {epoch_loss:.4f}")

    print("\nTransformer Fine-Tuning Complete!")

if __name__ == "__main__":
    main()