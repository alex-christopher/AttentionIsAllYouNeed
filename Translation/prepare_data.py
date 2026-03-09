import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
from tokenizer import FrenchTokenizer
from transformers import AutoTokenizer

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import tempfile
import shutil

def build_dataset_with_pyarrow(path_to_data_root, path_to_save, test_prop=0.005):
    
    temp_dir = tempfile.mkdtemp()
    parquet_files = []
    
    for dir in os.listdir(path_to_data_root):
        path_to_dir = os.path.join(path_to_data_root, dir)

        if os.path.isdir(path_to_dir):
            print("Processing : ", dir)

            french_file = english_file = None

            for text in os.listdir(path_to_dir):
                if text.endswith(".fr"):
                    french_file = os.path.join(path_to_dir, text)
                elif text.endswith(".en"):
                    english_file = os.path.join(path_to_dir, text)
            
            if french_file and english_file:
                with open(english_file, 'r', encoding='utf-8') as en_f, \
                     open(french_file, 'r', encoding='utf-8') as fr_f:
                    
                    en_batch = []
                    fr_batch = []
                    batch_size = 100000
                    
                    for en_line, fr_line in zip(en_f, fr_f):
                        en_batch.append(en_line.strip())
                        fr_batch.append(fr_line.strip())
                        
                        if len(en_batch) >= batch_size:
                            table = pa.Table.from_pydict({
                                'english_src': en_batch,
                                'french_tgt': fr_batch
                            })
                            
                            parquet_path = os.path.join(temp_dir, f"{dir}_{len(parquet_files)}.parquet")
                            pq.write_table(table, parquet_path)
                            parquet_files.append(parquet_path)
                            
                            en_batch = []
                            fr_batch = []
                    
                    if en_batch:
                        table = pa.Table.from_pydict({
                            'english_src': en_batch,
                            'french_tgt': fr_batch
                        })
                        parquet_path = os.path.join(temp_dir, f"{dir}_{len(parquet_files)}.parquet")
                        pq.write_table(table, parquet_path)
                        parquet_files.append(parquet_path)
    
    print("Loading all parquet files...")
    hf_dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    
    hf_dataset = hf_dataset.train_test_split(test_size=test_prop)
    hf_dataset.save_to_disk(path_to_save)
    
    shutil.rmtree(temp_dir)
    print(f"Dataset saved to {path_to_save}")

def tokenize_english2french_dataset(path_to_hf_data,
                                    path_to_save,
                                    num_workers=4,
                                    truncate=True,
                                    max_length=512,
                                    min_length=5):
    
    french_tokenizer = FrenchTokenizer(path_to_vocab="trained_tokenizer/french_wp.json",
                                       truncate=truncate,
                                       max_length=max_length)
    
    english_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    raw_dataset = load_from_disk(path_to_hf_data)

    def _tokenize_text(examples):

        english_text = examples["english_src"]
        french_text = examples["french_tgt"]

        src_ids = english_tokenizer(english_text, truncation=truncate, max_length=max_length)["input_ids"]
        tgt_ids = french_tokenizer.encode(french_text)

        batch= {"src_ids" : src_ids,
                "tgt_ids": tgt_ids}
        
        return batch
        
    tokenized_dataset = raw_dataset.map(_tokenize_text, batched=True, num_proc=num_workers)
    tokenized_dataset = tokenized_dataset.remove_columns(["english_src", "french_tgt"])

    filter_func = lambda examples: [len(e) > min_length for e in examples["tgt_ids"]]
    tokenized_dataset = tokenized_dataset.filter(filter_func, batched=True)

    print(tokenized_dataset)

    tokenized_dataset.save_to_disk(path_to_save)


if __name__ == "__main__":
    path_to_data_root = "E:/datasets/english2french"
    path_to_save_raw = "E:/datasets/english2french/hf_all_data"
    path_to_save_tok = "E:/datasets/english2french/hf_tokenized"

    # build_dataset_with_pyarrow(path_to_data_root=path_to_data_root, path_to_save=path_to_save_raw)

    tokenize_english2french_dataset(path_to_save_raw, path_to_save_tok)