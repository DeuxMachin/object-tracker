

import cv2
import numpy as np
from shared.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

class CameraHandler:
    """
    🎬 Clase principal para manejar tu cámara web
    
    Esta clase es como un "driver" que habla con tu cámara y nos da
    los frames de video que necesitamos para el seguimiento de objetos.
    """
    
    def __init__(self, camera_index: int = None):
        """
        🔧 Configura la cámara antes de usarla
        
        Args:
            camera_index: ¿Qué cámara usar? (0=primera, 1=segunda, etc.)
                         Si no especificas, usa la del archivo config.py
        """
        # Usar la cámara especificada o la de configuración por defecto
        self.camera_index = camera_index if camera_index is not None else CAMERA_INDEX
        self.width = CAMERA_WIDTH      # Ancho del video
        self.height = CAMERA_HEIGHT    # Alto del video  
        self.fps = CAMERA_FPS          # Frames por segundo
        self.cap = None                # Aquí guardaremos la conexión a la cámara
        
    def initialize(self) -> bool:
        """
        🚀 Enciende la cámara y la prepara para usarla
        
        Returns:
            True si todo salió bien, False si hubo problemas
        """
        try:
            # Intentar conectarse a la cámara
            print(f"🔍 Intentando conectar con cámara {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            # ¿Se pudo abrir la cámara?
            if not self.cap.isOpened():
                print(f"❌ No se pudo abrir la cámara {self.camera_index}")
                print("💡 Prueba cambiar CAMERA_INDEX en shared/config.py")
                return False
            
            # Configurar la calidad y tamaño del video
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            print(f"✅ ¡Cámara lista! Resolución: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"💥 Error inesperado al inicializar la cámara: {e}")
            return False
    
    def get_frame(self) -> tuple:
        """
        Captura un frame de la cámara.
        
        Returns:
            Tupla (success, frame)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """Libera la cámara."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Cámara liberada")
    
    def is_opened(self) -> bool:
        """Verifica si la cámara está abierta."""
        return self.cap is not None and self.cap.isOpened()


def find_available_cameras(max_cameras=10):
    """
    Busca cámaras disponibles en el sistema.
    
    Args:
        max_cameras: Número máximo de cámaras a probar
        
    Returns:
        Lista de índices de cámaras disponibles
    """
    available_cameras = []
    
    print("Buscando cámaras disponibles...")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"✓ Cámara encontrada en índice {i}")
            cap.release()
        else:
            # Probar diferentes backends
            for backend in [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_FFMPEG]:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(i)
                        print(f"✓ Cámara encontrada en índice {i} (backend alternativo)")
                        cap.release()
                        break
                cap.release()
    
    if not available_cameras:
        print("✗ No se encontraron cámaras disponibles")
    
    return available_cameras

def test_camera():
    """Función simple para probar la cámara."""
    # Buscar cámaras disponibles
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("❌ No hay cámaras disponibles en el sistema")
        print("💡 Verifica que:")
        print("  - Tu cámara esté conectada")
        print("  - No esté siendo usada por otra aplicación")
        print("  - Tengas permisos para acceder a la cámara")
        return False
    
    # Usar la primera cámara disponible
    camera_index = available_cameras[0]
    print(f"🎥 Usando cámara {camera_index}")
    
    # Intentar diferentes backends para mejor compatibilidad
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Media Foundation (Windows)"),
        (cv2.CAP_ANY, "Auto-detectar")
    ]
    
    cap = None
    backend_usado = None
    
    for backend, nombre in backends:
        print(f"🔧 Probando backend: {nombre}")
        cap = cv2.VideoCapture(camera_index, backend)
        
        if cap.isOpened():
            # Probar capturar un frame para verificar que funciona
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"✅ Backend {nombre} funcionando correctamente")
                backend_usado = nombre
                break
            else:
                print(f"⚠️ Backend {nombre} se abrió pero no captura frames")
                cap.release()
                cap = None
        else:
            print(f"❌ Backend {nombre} no pudo abrir la cámara")
            cap.release()
            cap = None
    
    if cap is None or not cap.isOpened():
        print("❌ No se pudo inicializar la cámara con ningún backend")
        return False

    # Configurar la cámara
    print(f"⚙️ Configurando cámara...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verificar configuración actual
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Configuración actual: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
    print(f"🔧 Backend usado: {backend_usado}")
    print("📹 Presiona 'q' en la ventana para salir")
    print()

    try:
        frame_count = 0
        frames_vacios = 0
        
        while True:
            ret, frame = cap.read()

            if not ret:
                print("⚠️ No se pudo leer frame (ret=False)")
                frames_vacios += 1
                if frames_vacios > 10:
                    print("❌ Demasiados frames fallidos, cerrando...")
                    break
                continue
                
            if frame is None:
                print("⚠️ Frame es None")
                frames_vacios += 1
                if frames_vacios > 10:
                    print("❌ Demasiados frames None, cerrando...")
                    break
                continue

            # Reset contador de frames vacíos
            frames_vacios = 0
            frame_count += 1

            # Debug cada 30 frames
            if frame_count % 30 == 0:
                print(f"📸 Frame {frame_count}: {frame.shape}, min={frame.min()}, max={frame.max()}")

            # Agregar información en el frame
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Backend: {backend_usado}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Presiona 'q' para salir", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Mostrar imagen
            cv2.imshow('Camera Test - Seguimiento de Objetos', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Saliendo por petición del usuario")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupción del usuario")
    except Exception as e:
        print(f"\n❌ Error durante la captura: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("🧹 Recursos liberados")
    
    return True
