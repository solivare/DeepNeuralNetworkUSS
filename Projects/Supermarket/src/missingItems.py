import os
import shutil
from glob import glob

# Ruta al dataset preparado
data_path = "../data/items_prepared"
train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "val")

# Clases que están en train pero no en val
clases_faltantes = {
    'Packages_Juice_Tropicana-Golden-Grapefruit',
    'Packages_Soy-Milk_Alpro-Fresh-Soy-Milk',
    'Packages_Juice_Tropicana-Juice-Smooth',
    'Fruit_Pear_Anjou',
    'Packages_Yoghurt_Arla-Natural-Mild-Low-Fat-Yoghurt',
    'Packages_Milk_Arla-Lactose-Medium-Fat-Milk',
    'Packages_Yoghurt_Arla-Natural-Yoghurt',
    'Fruit_Apple_Red-Delicious',
    'Vegetables_Tomato_Regular-Tomato',
    'Packages_Juice_Tropicana-Apple-Juice',
    'Vegetables_Potato_Solid-Potato',
    'Packages_Sour-Cream_Arla-Ecological-Sour-Cream',
    'Packages_Soy-Milk_Alpro-Shelf-Soy-Milk',
    'Packages_Sour-Milk_Arla-Sour-Milk',
    'Fruit_Pear_Kaiser',
    'Packages_Juice_Tropicana-Mandarin-Morning',
    'Vegetables_Pepper_Green-Bell-Pepper'
}

for clase in clases_faltantes:
    origen_dir = os.path.join(train_path, clase)
    destino_dir = os.path.join(val_path, clase)

    if not os.path.exists(origen_dir):
        print(f"⚠️ No existe: {origen_dir}")
        continue

    os.makedirs(destino_dir, exist_ok=True)

    # Buscar imágenes
    imagenes = glob(os.path.join(origen_dir, "*.jpg"))
    imagenes_a_mover = imagenes[:10]  # hasta 10

    for img_path in imagenes_a_mover:
        shutil.move(img_path, os.path.join(destino_dir, os.path.basename(img_path)))

    print(f"✅ Movidas {len(imagenes_a_mover)} imágenes de {clase}")