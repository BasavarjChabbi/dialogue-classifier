import pandas as pd
import sqlite3
import os

# A small sample dataset to make sure our pipeline works flawlessly

sample_data = {
    'text': [
        "Hello there!", "Good morning, how are you?", "Hey!", "Hi, glad to meet you.",
        "What time is it?", "How do I build a model?", "Where is the data stored?", "Can you explain this?",
        "I will finish this tomorrow.", "The weather is nice today.", "Deep learning is fascinating.", "Python is a great language.",
        "Yes, I completely agree.", "That sounds perfect to me.", "Okay, let's do that.", "I think you are right.",
        "Goodbye, see you later.", "Have a great day!", "Talk to you soon.", "Bye!"
    ],
    'label': [
        "Greeting", "Greeting", "Greeting", "Greeting",
        "Question", "Question", "Question", "Question",
        "Statement", "Statement", "Statement", "Statement",
        "Agreement", "Agreement", "Agreement", "Agreement",
        "Closing", "Closing", "Closing", "Closing"
    ]

    }

def setup_database(data_dict):
    # Convert our dictionary to a pandas DataFrame
    df = pd.DataFrame(data_dict)
    
    # Ensure the data folder exists
    os.makedirs('data', exist_ok=True)
    
    # Connect to SQLite (this creates the file if it doesn't exist)
    db_path = 'data/dialogue_db.sqlite'
    conn = sqlite3.connect(db_path)
    
    # Save the DataFrame as a SQL table
    df.to_sql('dialogue_acts', conn, if_exists='replace', index=False)
    conn.close()
    print(f"🎉 Success! Database created at: {db_path}")

if __name__ == "__main__":
    setup_database(sample_data)