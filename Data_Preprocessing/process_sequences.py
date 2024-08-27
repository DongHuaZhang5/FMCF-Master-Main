import os
import re
import pickle as pkl
import random

from data_process.utils import re_0001_, re_0002, re_opt
from data_process.xml_to_graph import xml_graph


def process_camel_snake_names(token):
    """Process camelCase and snake_case names."""
    if token.startswith("SimpleName") and "#" in token:
        parts = token.split("#")
        name_type = parts[0].lower()
        name_value = parts[1]
        if not re_opt.fullmatch(name_value):
            name_value = re_0001_.sub(re_0002, name_value).strip().lower()
        return [name_type, name_value]
    return [token.lower() if token not in ["$NUM$", "$STR$", "$ADDR$"] else f"「{token[1:-1]}」"]


def read_file(file_path):
    """Read a single line from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readline()


def get_contract_files(contract_folder):
    """Get a list of all files within a given contract folder."""
    return os.listdir(contract_folder)


def process_sbt_seq(contract_folder, seq_file):
    """Process the sequence of a smart contract."""
    seq_path = os.path.join("..", "contracts", "contracts_seqs_sbt", contract_folder, seq_file)
    seq = read_file(seq_path)
    tokens = seq.split()
    new_tokens = [token for token in tokens for token in process_camel_snake_names(token)]
    return " ".join(new_tokens)


def process_sbts_graphs_comms(min_comm_len, max_comm_len):
    """Process comments, sequences, graphs, and collect them into a dataset."""
    comm_contracts_path = os.path.join("..", "contracts", "comments_v11162020")
    contracts = get_contract_files(comm_contracts_path)
    total_contracts = len(contracts)

    dataset = []
    for idx, contract in enumerate(contracts):
        comm_files_path = os.path.join(comm_contracts_path, contract)
        files = get_contract_files(comm_files_path)

        for file in files:
            comm = read_file(os.path.join(comm_files_path, file))
            comm_tokens = comm.split()
            if min_comm_len <= len(comm_tokens) <= max_comm_len:
                sbt_seq = process_sbt_seq(contract, file.replace("_comm", ""))
                nodes, edges = xml_graph(contract, file.replace("_comm", ""))
                dataset.append((sbt_seq, nodes, edges, comm))
        print(f"{idx}/{total_contracts} finished!")

    save_path = os.path.join("..", "datasets", "smart_contracts", f"comms_{min_comm_len}_{max_comm_len}", "dataset.pkl")
    with open(save_path, "wb") as file:
        pkl.dump(dataset, file)


def split_dataset(dataset_name, test_prob, val_prob, min_comm_len=4, max_comm_len=20):
    """Split the dataset into training, validation, and testing sets."""
    load_path = os.path.join("..", "datasets", "smart_contracts", f"comms_{min_comm_len}_{max_comm_len}",
                             f"{dataset_name}.pkl")
    with open(load_path, "rb") as file:
        dataset = pkl.load(file)

    random.Random(345).shuffle(dataset)
    total_length = len(dataset)
    val_num = int(total_length * test_prob)
    test_num = int(total_length * val_prob)
    test_set = dataset[:test_num]
    val_set = dataset[test_num:test_num + val_num]
    train_set = dataset[test_num + val_num:]
    new_dataset = {'train': train_set, 'val': val_set, 'test': test_set}

    save_path = os.path.join("..", "datasets", "smart_contracts", f"comms_{min_comm_len}_{max_comm_len}",
                             "dataset_train_val_test.pkl")
    with open(save_path, "wb") as file:
        pkl.dump(new_dataset, file)


def refine_dataset(min_comm_len=4, max_comm_len=20):
    """Refine the dataset by selecting unique validation and test indices."""
    load_path = os.path.join("..", "datasets", "smart_contracts", f"comms_{min_comm_len}_{max_comm_len}",
                             "dataset_train_val_test.pkl")
    with open(load_path, "rb") as file:
        dataset = pkl.load(file)

    val_indices_path = os.path.join("..", "datasets", "smart_contracts", f"comms_{min_comm_len}_{max_comm_len}",
                                    "uniq_val_idics")
    with open(val_indices_path, "rb") as file:
        val_list = pkl.load(file)

    test_indices_path = os.path.join("..", "datasets", "smart_contracts", f"comms_{min_comm_len}_{max_comm_len}",
                                     "uniq_test_idics_x")
    with open(test_indices_path, "rb") as file:
        test_list = pkl.load(file)

    new_val_set = [dataset['val'][idx] for idx in val_list]
    new_test_set = [dataset['test'][idx] for idx in test_list]
    dataset.update({'val': new_val_set, 'test': new_test_set})

    print(dataset.keys())
    print(len(dataset['val']))

    save_path = os.path.join("..", "datasets", "smart_contracts", f"comms_{min_comm_len}_{max_comm_len}",
                             "dataset_train_val_test_uniq.pkl")
    with open(save_path, "wb") as file:
        pkl.dump(dataset, file)