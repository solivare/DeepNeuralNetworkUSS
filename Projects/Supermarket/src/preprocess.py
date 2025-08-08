import os
import shutil
from tqdm import tqdm
import yaml

def load_config(config_path="../config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def collect_images(base_dir):
    """
    Recolecta rutas de imágenes y sus clases completas por split
    """
    data = {"train": [], "val": [], "test": []}
    
    for split in data.keys():
        split_path = os.path.join(base_dir, split)
        for category in os.listdir(split_path):  # Fruit / Vegetables / Packages
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                continue
            for item in os.listdir(category_path):
                item_path = os.path.join(category_path, item)
                if not os.path.isdir(item_path):
                    continue
                for subitem in os.listdir(item_path):  # Subespecies o clases reales
                    sub_path = os.path.join(item_path, subitem)
                    if not os.path.isdir(sub_path):
                        continue
                    label = f"{category}_{item}_{subitem}"
                    for img_file in os.listdir(sub_path):
                        if img_file.lower().endswith(".jpg"):
                            full_path = os.path.join(sub_path, img_file)
                            data[split].append((full_path, label))
    return data

def organize_images(data, output_dir):
    """
    Copia imágenes en estructura compatible con flow_from_directory
    """
    for split, samples in data.items():
        print(f"📦 Organizando {split} ({len(samples)} imágenes)...")
        for img_path, label in tqdm(samples):
            target_folder = os.path.join(output_dir, split, label)
            os.makedirs(target_folder, exist_ok=True)
            filename = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(target_folder, filename))

def summarize_data(output_dir):
    """
    Muestra cuántas imágenes hay por clase y por partición
    """
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)
        print(f"\n📊 {split.upper()} SUMMARY:")
        total = 0
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            count = len(os.listdir(class_path))
            print(f"  {class_name}: {count}")
            total += count
        print(f"  Total imágenes: {total}")

def main():
    config = load_config()
    base_dir = config["paths"]["raw_data"]
    output_dir = config["paths"]["prepared_data"]

    data = collect_images(base_dir)
    organize_images(data, output_dir)
    summarize_data(output_dir)
    print("\n✅ Preprocesamiento completo.")

if __name__ == "__main__":
    main()