import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_dataset(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def calculate_statistics(data, label_key):
    sbts_len = [len(item[label_key].split(" ")) for item in data]
    stats = {
        "mean": np.mean(sbts_len),
        "median": np.median(sbts_len),
        "mode": np.argmax(np.bincount(sbts_len))
    }
    print(f"mean: {stats['mean']}")
    print(f"median: {stats['median']}")
    print(f"mode: {stats['mode']}")

    count_dict = defaultdict(list)
    for length in sbts_len:
        for key in count_dict.keys():
            if key.startswith(str(length)) or (key.endswith('-') and length >= int(key[:-1])):
                count_dict[key].append(length)
                break

    for key, values in count_dict.items():
        print(f"{key}: {len(values) / len(data)}")

def check_dataset():
    base_path = Path("../datasets/smart_contracts/comms_4_20")
    datasets = [
        "dataset_train_val_test.pkl",
        "dataset_train_val_test_uniq.pkl"
    ]

    for dataset in datasets:
        data = load_dataset(base_path / dataset)
        for key in data.keys():
            if key != 'train':  # Assuming 'train' is always present
                print(f"{key}: {len(data[key])}")

