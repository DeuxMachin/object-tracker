import sys
import os

# Agregar el directorio src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.client.camera_handler import test_camera

if __name__ == "__main__":
    print("=== Test de Cámara Web ===")
    print("Iniciando prueba de cámara...")
    test_camera()
