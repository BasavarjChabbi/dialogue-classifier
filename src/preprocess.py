import pandas as pd
import sqlite3
import os

# A small sample dataset to make sure our pipeline works flawlessly
sample_data = {
    'text': [
        "Hello, how are you?", 
        "I am doing well, thanks.", 
        "What is the weather like?", 
        "It is sunny today.", 
        "Goodbye!"
    ],
    'label': ["Greeting", "Statement", "Question", "Statement", "Closing"]
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