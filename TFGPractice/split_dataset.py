
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = r"c:\TFG\TFGPractice\CKMImagenet"
MANIFEST_PATH = os.path.join(BASE_DIR, "Datasets", "ckm_dataset_manifest.csv")
TRAIN_OUTPUT = os.path.join(BASE_DIR, "Datasets", "train_manifest.csv")
TEST_OUTPUT = os.path.join(BASE_DIR, "Datasets", "test_manifest.csv")

def split_dataset():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    print(f"Reading manifest from {MANIFEST_PATH}...")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Shuffle and split 80/20
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)
    
    print(f"Saved train manifest to: {TRAIN_OUTPUT}")
    print(f"Saved test manifest to: {TEST_OUTPUT}")

if __name__ == "__main__":
    split_dataset()
