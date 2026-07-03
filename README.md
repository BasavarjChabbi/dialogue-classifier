# Context-Aware Dialogue Act Classification Engine

An end-to-end Deep Learning pipeline built using PyTorch to classify conversational intents (Dialogue Acts) from text transcripts. The project features a structured local SQL data pipeline, word vector tokenization handling, and a Bidirectional LSTM neural network capable of extracting sequence features simultaneously from both forward and backward contexts.

## 🚀 Key Features
* **SQL Data Storage:** Integrates structured local SQLite databases to simulate scalable production enterprise data configurations.
* **Custom PyTorch Dataset Pipeline:** Implements low-level dynamic text tokenization, vocabulary mapping, and dynamic batch padding (`collate_fn`) to handle sentences of variable lengths efficiently.
* **Deep Learning Engine:** Features a **Bidirectional LSTM** architecture to model context and sequence dynamics across human conversations.
* **Interactive Inference CLI:** A command-line wrapper enabling real-time classification evaluation.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`)
* **Data Management:** SQLite, Pandas
* **Evaluation Metrics:** Scikit-Learn

---

## 📁 Repository Structure
```text
dialogue-classifier/
├── data/
│   └── dialogue_db.sqlite    # Generated SQLite Database
├── src/
│   ├── preprocess.py         # DB Setup & Mock Data Pipeline
│   ├── dataset.py            # Tokenizer & PyTorch DataLoader
│   ├── train.py              # Bi-LSTM Model Architecture & Training Routine
│   └── predict.py            # Interactive CLI Inference Loop
├── requirements.txt          # Frozen Environment Dependencies
└── README.md                 # Production Documentation