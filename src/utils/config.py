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
CAMERA_WIDTH = 640   # Ancho en píxeles
CAMERA_HEIGHT = 480  # Alto en píxeles

# Frames por segundo - más alto = más fluido pero usa más recursos
CAMERA_FPS = 30
