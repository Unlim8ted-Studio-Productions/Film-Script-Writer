import os
from datasets import Dataset

def load_scripts(directory):
    scripts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                scripts.append({"text": text})
    return scripts

def save_dataset(scripts, output_path):
    dataset = Dataset.from_list(scripts)
    dataset.save_to_disk(output_path)

if __name__ == "__main__":
    scripts = load_scripts('scripts')
    save_dataset(scripts, 'dataset')
