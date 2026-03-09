from pathlib import Path
import pandas as pd
import random
import math
import hashlib
import os

def load_metadata_cache(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find metadata cache: {csv_path}")
    print(f"Loading metadata from csv: {csv_path}")
    return pd.read_csv(csv_path)

def normalize_session_name(x):
    x = str(x).replace("\\", "/")
    return Path(x).stem

def list_sessions_in_user_folder(user, data_root):
    user_dir = Path(data_root) / str(user)
    if not user_dir.exists():
        print(f"Warning: folder does not exist for user {user}: {user_dir}")
        return set()

    sessions = set()
    for p in user_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}:
            sessions.add(p.stem)

    return sessions

def stable_user_seed(base_seed, user):
    s = f"{base_seed}_{user}".encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)

def split_sessions(sessions, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42, mode="random"):
    total = train_frac + val_frac + test_frac
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    sessions = list(sessions)

    if mode == "chronological":
        sessions = sorted(sessions)
    elif mode == "random":
        rng = random.Random(seed)
        rng.shuffle(sessions)
    else:
        raise ValueError("mode must be 'random' or 'chronological'")

    n = len(sessions)
    if n == 0:
        return [], [], []
    if n == 1:
        return sessions, [], []
    if n == 2:
        return [sessions[0]], [], [sessions[1]]

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    if n_train >= n:
        n_train = n - 1
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    if val_frac > 0 and n >= 3 and n_val == 0:
        if n_train > 1:
            n_train -= 1
            n_val = 1

    train = sessions[:n_train]
    val = sessions[n_train:n_train + n_val]
    test = sessions[n_train + n_val:]

    return train, val, test

def render_yaml_text(user_label, dataset_dict):
    lines = [
        "# @package _global_",
        f"user: {user_label}",
        "dataset:"
    ]

    for split in ["train", "val", "test"]:
        items = dataset_dict.get(split, [])
        if not items:
            lines.append(f"  {split}: []")
            continue

        lines.append(f"  {split}:")
        for item in items:
            lines.append(f"  - user: {item['user']}")
            lines.append(f"    session: {item['session']}")

    return "\n".join(lines) + "\n"

def make_dataset_yaml(
    users,
    data_root,
    output_path,
    metadata_csv,
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1,
    split_mode="random",
    seed=42,
    require_existing_files=True,
    user_label=None
):
    metadata = load_metadata_cache(metadata_csv).copy()
    metadata["user"] = metadata["user"].astype(str)
    metadata["session"] = metadata["session"].astype(str)

    users = [str(u) for u in users]

    if user_label is None:
        user_label = "single_user" if len(users) == 1 else "multi_user"

    dataset = {"train": [], "val": [], "test": []}

    print(f"Building YAML for users: {users}")
    print(f"Fractions: train={train_frac}, val={val_frac}, test={test_frac}")
    print(f"Split mode: {split_mode}")

    for user in users:
        user_rows = metadata.loc[metadata["user"] == user, ["user", "session"]].drop_duplicates()
        meta_sessions = set(user_rows["session"].map(normalize_session_name))

        print(f"\nUser {user}")
        print(f"  Sessions in metadata: {len(meta_sessions)}")

        if require_existing_files:
            folder_sessions = list_sessions_in_user_folder(user, data_root)
            print(f"  Sessions found in folder: {len(folder_sessions)}")
            sessions = sorted(meta_sessions & folder_sessions)
            print(f"  Sessions kept after intersection: {len(sessions)}")
        else:
            sessions = sorted(meta_sessions)
            print(f"  Sessions kept: {len(sessions)}")

        if len(sessions) == 0:
            print(f"  Warning: no sessions available for user {user}")
            continue

        user_seed = stable_user_seed(seed, user)
        train_s, val_s, test_s = split_sessions(
            sessions,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=user_seed,
            mode=split_mode
        )

        print(f"  train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

        for s in train_s:
            dataset["train"].append({"user": user, "session": s})
        for s in val_s:
            dataset["val"].append({"user": user, "session": s})
        for s in test_s:
            dataset["test"].append({"user": user, "session": s})

    yaml_text = render_yaml_text(user_label, dataset)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_text, encoding="utf-8")

    print(f"\nSaved YAML to: {output_path}")
    print(f"Total train: {len(dataset['train'])}")
    print(f"Total val:   {len(dataset['val'])}")
    print(f"Total test:  {len(dataset['test'])}")

    return yaml_text, dataset