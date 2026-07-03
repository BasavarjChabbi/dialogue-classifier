import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class DialogueDataset(Dataset):
    def __init__(self, db_path):
        # 1. Load data from the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text, label FROM dialogue_acts")
        rows = cursor.fetchall()
        conn.close()
        
        self.raw_texts = [row[0] for row in rows]
        self.raw_labels = [row[1] for row in rows]
        
        # 2. Build a simple word-to-index vocabulary
        all_words = " ".join(self.raw_texts).lower().replace("?", "").replace(".", "").split()
        word_counts = Counter(all_words)
        
        # Reserved tokens: 0 for padding, 1 for unknown words
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for word in word_counts:
            self.vocab[word] = len(self.vocab)
            
        # 3. Build a label-to-index mapping
        unique_labels = list(set(self.raw_labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        # 4. Convert all data to numerical sequences
        self.encoded_texts = [self._tokenize(text) for text in self.raw_texts]
        self.encoded_labels = [self.label_map[label] for label in self.raw_labels]

    def _tokenize(self, text):
        cleaned = text.lower().replace("?", "").replace(".", "").split()
        return [self.vocab.get(word, 1) for word in cleaned]

    def __len__(self):
        return len(self.raw_texts)

    def __getitem__(self, idx):
        # Return text sequence as a tensor and label as a tensor
        return torch.tensor(self.encoded_texts[idx], dtype=torch.long), torch.tensor(self.encoded_labels[idx], dtype=torch.long)

# Custom collate function to handle variable length sentences via padding
def collate_fn(batch):
    texts, labels = zip(*batch)
    # Pad sequences automatically with 0 (<PAD>)
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_texts, labels

if __name__ == "__main__":
    # Test our dataset pipeline
    dataset = DialogueDataset('data/dialogue_db.sqlite')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    print("Vocabulary Size:", len(dataset.vocab))
    print("Label Mapping:", dataset.label_map)
    
    # Grab one batch to verify shapes
    for batch_text, batch_label in dataloader:
        print("\nPadded Batch Tensor Shape (Batch Size, Sequence Length):", batch_text.shape)
        print("Labels Tensor:", batch_label)
        break