import random
import json
import os

def add_and_split_new_data(old_data, new_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    all_existing_paths = set(old_data["train"].keys()) | set(old_data["val"].keys()) | set(old_data["test"].keys())
    non_overlapping_data = {img: label for img, label in new_data.items() if img not in all_existing_paths}

    non_overlapping_items = list(non_overlapping_data.items())
    random.shuffle(non_overlapping_items)

    total_new = len(non_overlapping_items)
    train_end = int(total_new * train_ratio)
    val_end = train_end + int(total_new * val_ratio)

    for idx, (image_path, label) in enumerate(non_overlapping_items):
        if idx < train_end:
            old_data["train"][image_path] = label
        elif idx < val_end:
            old_data["val"][image_path] = label
        else:
            old_data["test"][image_path] = label

    return old_data


def read_json(input_path):
    try:
        with open(input_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return {}


def save_json(data, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    base_dir = '/teamspace/studios/this_studio/dataset'
    augmented_path = os.path.join(base_dir, 'augmented_split.json')
    original_path = os.path.join(base_dir, 'split.json')
    output_path = os.path.join(base_dir, 'processed_augmented_split.json')

    augmented_split = read_json(augmented_path)
    original_split = read_json(original_path)

    augmented_processed_split = add_and_split_new_data(original_split, augmented_split)

    save_json(augmented_processed_split, output_path)
