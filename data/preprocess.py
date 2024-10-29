import os
import json
import random


def read_json(input_path):
    try:
        with open(input_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return {}


def get_labeled_subset(data_dir):
    labeled_data_dir = os.path.join(data_dir, 'example_set')
    labels = ['aligned', 'not_aligned']
    
    images_dict = {
        os.path.join(root, file): label
        for label in labels
        for root, _, files in os.walk(os.path.join(labeled_data_dir, label))
        for file in files
    }
    return images_dict


def get_labeled_trainset(labeled_dict, unlabeled_data_dir):
    return {
        os.path.join(unlabeled_data_dir, img_name): label
        for img_name, label in labeled_dict.items()
    }


def stratified_split(data, train_ratio=0.7, val_ratio=0.15):
    def split_dict(data_dict):
        keys = list(data_dict.keys())
        random.shuffle(keys)
        
        train_end = int(len(keys) * train_ratio)
        val_end = train_end + int(len(keys) * val_ratio)
        
        return (
            {k: data_dict[k] for k in keys[:train_end]},
            {k: data_dict[k] for k in keys[train_end:val_end]},
            {k: data_dict[k] for k in keys[val_end:]}
        )

    aligned = {k: v for k, v in data.items() if v == "aligned"}
    not_aligned = {k: v for k, v in data.items() if v == "not_aligned"}

    train_aligned, val_aligned, test_aligned = split_dict(aligned)
    train_not_aligned, val_not_aligned, test_not_aligned = split_dict(not_aligned)

    return (
        {**train_aligned, **train_not_aligned},
        {**val_aligned, **val_not_aligned},
        {**test_aligned, **test_not_aligned}
    )


if __name__ == '__main__':
    data_dir = '/teamspace/s3_connections/dtp-sbm-segmentation-video-tasks-bars-stopper-alignment-images-hackaton-usi'
    unlabeled_data_dir = os.path.join(data_dir, 'train_set')

    labeled_imgs = get_labeled_subset(data_dir)

    good_light_labeled_dict = read_json('dataset/label_extraction/good_light_image_labels.json')
    bad_light_labeled_dict = read_json('dataset/label_extraction/bad_light_image_label.json')

    labeled_data = {
        **labeled_imgs,
        **get_labeled_trainset(good_light_labeled_dict, unlabeled_data_dir),
        **get_labeled_trainset(bad_light_labeled_dict, unlabeled_data_dir)
    }

    train_set, val_set, test_set = stratified_split(labeled_data)

    output_data = {'train': train_set, 'val': val_set, 'test': test_set}
    with open('dataset/split.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
