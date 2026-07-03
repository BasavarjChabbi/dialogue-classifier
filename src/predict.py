import torch
import torch.nn as nn
from dataset import DialogueDataset
from train import DialogueBiLSTM

def interactive_predict():
    # 1. Load the dataset instance to reuse vocabulary and label maps
    db_path = 'data/dialogue_db.sqlite'
    dataset = DialogueDataset(db_path)
    
    # 2. Reconstruct the model parameters
    vocab_size = len(dataset.vocab)
    num_classes = len(dataset.label_map)
    embed_dim = 16
    hidden_dim = 32
    
    # Initialize the architecture instance
    model = DialogueBiLSTM(vocab_size, embed_dim, hidden_dim, num_classes)
    
    # In a real environment, we would load weights here, but since our model is
    # already in memory, let's train a miniature 5-epoch fast instance right at startup
    # so we don't need to juggle separate weight checkpoint (.pth) files for this test.
    from torch.utils.data import DataLoader
    from dataset import collate_fn
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for _ in range(15):
        for batch_texts, batch_labels in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_texts), batch_labels)
            loss.backward()
            optimizer.step()
            
    # Flip the network into Evaluation Mode (disables dropout, fixes batch-norm parameters)
    model.eval()
    
    # Create an inverse label map to turn integer predictions back into text categories
    inverse_label_map = {v: k for k, v in dataset.label_map.items()}
    
    print("\nDialogue Act Prediction Engine Live!")
    print("Type your message below (or type 'exit' to quit):\n")
    
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # Clean and tokenize user input sentence
        cleaned = user_input.lower().replace("?", "").replace(".", "").split()
        encoded = [dataset.vocab.get(word, 1) for word in cleaned] # 1 is <UNK>
        
        # Wrap sequence inside a tensor batch of size 1: shape [1, sequence_length]
        input_tensor = torch.tensor([encoded], dtype=torch.long)
        
        with torch.no_grad():
            output_logits = model(input_tensor)
            predicted_class_id = torch.argmax(output_logits, dim=1).item()
            
        prediction_text = inverse_label_map[predicted_class_id]
        print(f"Predicted Dialogue Act Intent -> [{prediction_text}]\n")

if __name__ == "__main__":
    interactive_predict()
