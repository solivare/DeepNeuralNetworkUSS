from pathlib import Path
import yaml

"""
CHANGE v2:
- Lee run_id desde el YAML.
- No hace falta tocar este archivo cuando se inicie una nueva versión, solo config.yaml.
"""

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]           # Projects/Credit
DATA_DIR = BASE_DIR / "data"

# Leer run_id directamente del YAML
with open(BASE_DIR / "config.yaml") as f:                 # CHANGE v2
    RUN_ID = yaml.safe_load(f)["run_id"]                 # CHANGE v2

# Carpetas de salida parametrizadas
MODELS_DIR  = BASE_DIR / "models"  / RUN_ID
REPORTS_DIR = BASE_DIR / "reports" / RUN_ID
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Debug
if __name__ == "__main__":                         
    print("RUN_ID     :", RUN_ID)
    print("MODELS_DIR :", MODELS_DIR)
    print("REPORTS_DIR:", REPORTS_DIR)