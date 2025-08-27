import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Ruta al dataset original
BASE_DIR = "/Users/saolivap/WorkArea/samples/BreastCancer"

# Directorio donde se almacenarán las imágenes organizadas
OUTPUT_DIR = "../data/breast_images_prepared"

# Proporción de entrenamiento
TEST_SIZE = 0.2
RANDOM_STATE = 42


def collect_images_by_class(base_dir, max_per_class=None):
    image_paths = []
    labels = []

    class_counts = {'0': 0, '1': 0}

    for patient_id in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_id)

        if not os.path.isdir(patient_path):
            continue

        for class_label in ['0', '1']:
            class_path = os.path.join(patient_path, class_label)

            if not os.path.isdir(class_path):
                continue

            if max_per_class and class_counts[class_label] >= max_per_class:
                continue

            for filename in os.listdir(class_path):
                if not filename.endswith('.png'):
                    continue

                if max_per_class and class_counts[class_label] >= max_per_class:
                    break

                image_paths.append(os.path.join(class_path, filename))
                labels.append(int(class_label))
                class_counts[class_label] += 1

    return image_paths, labels


def split_and_organize(image_paths, labels, output_dir):
    print("📦 Repartiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    sets = [("train", X_train, y_train), ("test", X_test, y_test)]

    for split_name, images, y_vals in sets:
        for img_path, label in tqdm(zip(images, y_vals), desc=f"Organizando {split_name}"):
            target_dir = os.path.join(output_dir, split_name, str(label))
            os.makedirs(target_dir, exist_ok=True)
            filename = os.path.basename(img_path)
            dest_path = os.path.join(target_dir, filename)
            shutil.copy(img_path, dest_path)


def summarize_dataset(output_dir):
    print("\n📊 Resumen de imágenes por clase:")
    for split in ['train', 'test']:
        print(f"\n🔹 {split.upper()}")
        for cls in ['0', '1']:
            folder = os.path.join(output_dir, split, cls)
            if os.path.exists(folder):
                count = len(os.listdir(folder))
                print(f"  Clase {cls}: {count} imágenes")


def main():
    from yaml import safe_load
    with open("../config.yaml", "r") as f:
        config = safe_load(f)

    max_per_class = config.get("preprocessing", {}).get("max_images_per_class", None)
    print(f"🔍 Recolectando imágenes (máximo por clase: {max_per_class})...")
    image_paths, labels = collect_images_by_class(BASE_DIR, max_per_class)

    print(f"Total imágenes seleccionadas: {len(image_paths)}")
    print("🗂️ Organizando imágenes en carpetas...")
    split_and_organize(image_paths, labels, OUTPUT_DIR)

    summarize_dataset(OUTPUT_DIR)
    print("\n✅ Preprocesamiento completado.")


if __name__ == "__main__":
    main()