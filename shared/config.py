"""
Configuración básica para la cámara web.

Este archivo contiene las configuraciones principales que puedes ajustar
según tu cámara y preferencias. Si tu cámara no funciona, prueba cambiando
el CAMERA_INDEX (0, 1, 2, etc.).
"""

# ===== CONFIGURACIÓN DE CÁMARA =====
# Si tu cámara no funciona, prueba cambiar este número (0, 1, 2, etc.)
CAMERA_INDEX = 0

# Resolución del video - ajusta según tu cámara y rendimiento deseado
# NOTA: Estas configuraciones serán optimizadas después de ejecutar full_hd_validator.py
CAMERA_WIDTH = 1920  # Full HD para pruebas de rendimiento
CAMERA_HEIGHT = 1080 # Full HD para pruebas de rendimiento

# Frames por segundo - más alto = más fluido pero usa más recursos
CAMERA_FPS = 30      # FPS máximo para pruebas

# ===== CONFIGURACIÓN DE RED =====
# URL del servidor donde enviar los frames
SERVER_URL = "http://192.168.1.84:5000/upload"

# Configuración de streaming para cliente-servidor
# NOTA: Estos valores serán optimizados después de las pruebas de rendimiento
STREAMING_FPS = 25          # FPS para transmisión (se optimizará según resultados)
JPEG_QUALITY = 80           # Calidad JPEG (se optimizará según resultados)
RECONNECT_DELAY = 5         # Segundos entre intentos de reconexión
MAX_RETRIES = 3             # Máximo intentos de envío por frame

# Modo de operación
MODE_LOCAL = 'local'        # Para testing con Flask local
MODE_REMOTE = 'remote'      # Para envío al servidor remoto
DEFAULT_MODE = MODE_REMOTE  # Modo por defecto
