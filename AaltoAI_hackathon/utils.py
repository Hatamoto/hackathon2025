import os
import json
import hashlib


def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def has_file_changed(file_path, hash_store_path="data/file_hashes.json"):
    # Load previous hashes
    if os.path.exists(hash_store_path):
        with open(hash_store_path, "r") as f:
            hash_store = json.load(f)
    else:
        hash_store = {}

    current_hash = compute_file_hash(file_path)
    previous_hash = hash_store.get(file_path)

    if current_hash != previous_hash:
        # Update hash and return True (i.e., file changed)
        hash_store[file_path] = current_hash
        with open(hash_store_path, "w") as f:
            json.dump(hash_store, f, indent=2)
        return True
    return False
