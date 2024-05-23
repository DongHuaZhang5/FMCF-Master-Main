import hashlib
import os

def hash_file_contents(file_path):
    """
    Generate a hash for the contents of a file.
    :param file_path: Path to the file
    :return: Hash string
    """
    hasher = hashlib.sha256()
    with open(file_path, 'r', encoding='utf-8') as file:
        hasher.update(file.read().encode('utf-8'))
    return hasher.hexdigest()

def remove_duplicates(dataset_path):
    """
    Remove duplicate files based on their content hashes.
    :param dataset_path: Path to the dataset directory
    """
    files = os.listdir(dataset_path)
    hashes = {}
    for file in files:
        file_path = os.path.join(dataset_path, file)
        file_hash = hash_file_contents(file_path)
        if file_hash in hashes:
            # If hash is found in dictionary, delete the duplicate file
            os.remove(file_path)
        else:
            # Otherwise, add the hash to dictionary
            hashes[file_hash] = file_path

def process_datasets(dataset1_path, dataset2_path):
    """
    Process two datasets and remove redundant entries.
    :param dataset1_path: Path to the first dataset directory
    :param dataset2_path: Path to the second dataset directory
    """
    # Process each dataset initially
    remove_duplicates(dataset1_path)
    remove_duplicates(dataset2_path)

    # Additional logic could be added here if files need to be merged or compared across datasets

# Example usage
process_datasets("../dataset1", "../dataset2")
