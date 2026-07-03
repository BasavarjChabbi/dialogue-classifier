import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer

class TransformerDialogueDataset(Dataset):
    def __init__(self, db_path):
        # 1. Load data from SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text, label FROM dialogue_acts")
        rows = cursor.fetchall()
        conn.close()
        
        self.raw_texts = [row[0] for row in rows]
        self.raw_labels = [row[1] for row in rows]
        
        # 2. Load the pre-trained DistilBERT Tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # 3. Create label mapping
        unique_labels = list(set(self.raw_labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.encoded_labels = [self.label_map[label] for label in self.raw_labels]

    def __len__(self):
        return len(self.raw_texts)

    def __getitem__(self, idx):
        text = self.raw_texts[idx]
        label = self.encoded_labels[idx]
        
        # Tokenize using Hugging Face constraints
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=16, # Keeps tensor matrices small and fast
            return_tensors="pt"
        )
        
        # Return input IDs, Attention Mask, and target Label
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    dataset = TransformerDialogueDataset('data/dialogue_db.sqlite')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch in dataloader:
        print("Input IDs Shape:", batch['input_ids'].shape)
        print("Attention Mask Shape:", batch['attention_mask'].shape)
        print("Labels Tensor:", batch['label'])
        break