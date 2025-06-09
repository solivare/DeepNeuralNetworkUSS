#!/bin/bash

echo "Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

echo "Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Entorno listo. Usa 'source venv/bin/activate' para activarlo."