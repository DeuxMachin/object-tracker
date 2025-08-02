"""
Modo simulación para desarrollo sin cámara física.
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import CAMERA_WIDTH, CAMERA_HEIGHT

class SimulatedCamera:
    """Cámara simulada para desarrollo sin hardware."""
    
    def __init__(self):
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.frame_count = 0
        self.is_active = False
        
    def initialize(self):
        """Inicializa la cámara simulada."""
        self.is_active = True
        return True
    
    def get_frame(self):
        """Genera un frame simulado."""
        if not self.is_active:
            return False, None
        
        # Crear frame sintético
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Fondo degradado
        for y in range(self.height):
            for x in range(self.width):
                frame[y, x] = [
                    int(255 * x / self.width),  # Rojo
                    int(255 * y / self.height), # Verde
                    128  # Azul fijo
                ]
        
        # Círculo que se mueve
        center_x = int(self.width/2 + 100 * np.sin(self.frame_count * 0.05))
        center_y = int(self.height/2 + 50 * np.cos(self.frame_count * 0.03))
        
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        cv2.circle(frame, (center_x, center_y), 30, (0, 0, 0), 2)
        
        # Texto informativo
        cv2.putText(frame, "MODO SIMULACION", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Objeto simulado para tracking", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Presiona 'q' para salir", (10, self.height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.frame_count += 1
        return True, frame
    
    def release(self):
        """Libera la cámara simulada."""
        self.is_active = False
    
    def is_opened(self):
        """Verifica si la cámara está activa."""
        return self.is_active

def test_simulation():
    """Test con cámara simulada."""
    print("🎭 Iniciando simulación...")
    
    camera = SimulatedCamera()
    
    if not camera.initialize():
        print("❌ Error inicializando simulación")
        return False
    
    print("📹 Presiona 'q' para salir")
    
    try:
        while True:
            ret, frame = camera.get_frame()
            
            if not ret:
                break
            
            cv2.imshow('🎭 Cámara Simulada - Desarrollo', frame)
            
            # Salir con 'q'
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupción del usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    return True

def main():
    """Función principal."""
    print("="*50)
    print("🎭 MODO SIMULACIÓN - DESARROLLO SIN CÁMARA")
    print("="*50)
    print("Este modo permite desarrollar sin cámara física")
    print("Útil para WSL o sistemas sin acceso a hardware")
    print()
    
    success = test_simulation()
    
    if success:
        print("\n✅ Simulación completada")
    else:
        print("\n❌ Error en la simulación")
    

if __name__ == "__main__":
    main()
