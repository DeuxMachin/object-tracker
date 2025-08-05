"""
MJPEG Stream Client - Cliente con Streaming MJPEG Continuo

Implementa streaming continuo MJPEG para eliminar overhead HTTP:
- Stream multipart/x-mixed-replace continuo
- Sin confirmaciones HTTP individuales por frame
- Menor latencia y mayor throughput
- Compatible con visualización directa en navegadores

Ejecutar: python mjpeg_stream_client.py
"""

import cv2
import time
import threading
import sys
import os
import socket
import struct
from io import BytesIO

# Agregar directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.camera_handler import CameraHandler
from shared.config import *

class MJPEGStreamClient:
    """
    Cliente de streaming MJPEG continuo.
    
    Envía video como stream MJPEG multipart continuo,
    eliminando el overhead de HTTP requests individuales.
    """
    
    def __init__(self, server_host=None, server_port=5001, target_fps=None, jpeg_quality=None):
        """
        Inicializa el cliente MJPEG.
        
        Args:
            server_host: IP del servidor
            server_port: Puerto para MJPEG (diferente del HTTP)
            target_fps: FPS objetivo
            jpeg_quality: Calidad JPEG
        """
        # Extraer host de SERVER_URL si no se especifica
        if server_host is None:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(SERVER_URL)
                server_host = parsed.hostname or "192.168.1.84"
            except:
                server_host = "192.168.1.84"
        
        self.server_host = server_host
        self.server_port = server_port
        self.target_fps = target_fps or STREAMING_FPS
        self.jpeg_quality = jpeg_quality or JPEG_QUALITY
        self.frame_interval = 1.0 / self.target_fps
        
        # Componentes
        self.camera = CameraHandler()
        self.socket = None
        self.running = False
        self.stream_thread = None
        
        # Estadísticas
        self.stats = {
            'frames_sent': 0,
            'frames_failed': 0,
            'bytes_sent': 0,
            'start_time': None,
            'connection_errors': 0,
            'reconnections': 0
        }
        
        print("📺 MJPEGStreamClient inicializado")
        print(f"📊 Configuración:")
        print(f"   Servidor: {self.server_host}:{self.server_port}")
        print(f"   Target FPS: {self.target_fps}")
        print(f"   Calidad JPEG: {self.jpeg_quality}%")
        print(f"   Protocolo: MJPEG Stream continuo")
    
    def initialize(self):
        """Inicializa el sistema."""
        print("🔧 Inicializando cliente MJPEG...")
        
        # Inicializar cámara
        if not self.camera.initialize():
            print("❌ Error inicializando cámara")
            return False
        
        # Probar conexión al servidor
        if not self.test_server_connection():
            print("❌ Error conectando al servidor MJPEG")
            return False
        
        print("✅ Inicialización exitosa")
        return True
    
    def test_server_connection(self):
        """Prueba conexión al servidor MJPEG."""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(5.0)
            result = test_socket.connect_ex((self.server_host, self.server_port))
            test_socket.close()
            
            if result == 0:
                print(f"✅ Servidor MJPEG accesible en {self.server_host}:{self.server_port}")
                return True
            else:
                print(f"❌ No se puede conectar a {self.server_host}:{self.server_port}")
                print(f"💡 Asegúrate de que el servidor MJPEG esté corriendo")
                return False
                
        except Exception as e:
            print(f"❌ Error probando conexión: {e}")
            return False
    
    def connect_to_server(self):
        """Establece conexión socket con el servidor."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)
            self.socket.connect((self.server_host, self.server_port))
            
            # Enviar headers MJPEG
            headers = (
                f"POST /mjpeg_stream HTTP/1.1\\r\\n"
                f"Host: {self.server_host}:{self.server_port}\\r\\n"
                f"Content-Type: multipart/x-mixed-replace; boundary=mjpegboundary\\r\\n"
                f"Connection: keep-alive\\r\\n"
                f"\\r\\n"
            ).encode('utf-8')
            
            self.socket.sendall(headers)
            print(f"✅ Conectado al servidor MJPEG")
            return True
            
        except Exception as e:
            print(f"❌ Error conectando: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def send_mjpeg_frame(self, frame_data):
        """Envía frame como parte del stream MJPEG."""
        try:
            # Formato multipart MJPEG
            boundary = b"--mjpegboundary\\r\\n"
            content_type = b"Content-Type: image/jpeg\\r\\n"
            content_length = f"Content-Length: {len(frame_data)}\\r\\n".encode('utf-8')
            headers_end = b"\\r\\n"
            frame_end = b"\\r\\n"
            
            # Construir frame completo
            mjpeg_frame = boundary + content_type + content_length + headers_end + frame_data + frame_end
            
            # Enviar frame
            self.socket.sendall(mjpeg_frame)
            return True
            
        except socket.error as e:
            print(f"❌ Error enviando frame: {e}")
            return False
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return False
    
    def compress_frame(self, frame):
        """Comprime frame a JPEG."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, encoded_frame = cv2.imencode('.jpg', frame, encode_params)
        
        if not success:
            return None
        return encoded_frame.tobytes()
    
    def streaming_loop(self):
        """Loop principal de streaming MJPEG."""
        print("🎬 Iniciando stream MJPEG...")
        self.stats['start_time'] = time.time()
        last_frame_time = 0
        last_stats_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Control de tasa de frames
            if current_time - last_frame_time < self.frame_interval:
                time.sleep(0.001)
                continue
            
            # Capturar frame
            ret, frame = self.camera.get_frame()
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            # Comprimir frame
            compressed_frame = self.compress_frame(frame)
            if compressed_frame is None:
                continue
            
            # Verificar conexión
            if self.socket is None:
                if not self.connect_to_server():
                    self.stats['connection_errors'] += 1
                    time.sleep(1.0)
                    continue
            
            # Enviar frame MJPEG
            success = self.send_mjpeg_frame(compressed_frame)
            
            if success:
                self.stats['frames_sent'] += 1
                self.stats['bytes_sent'] += len(compressed_frame)
            else:
                self.stats['frames_failed'] += 1
                # Reconectar en caso de error
                if self.socket:
                    self.socket.close()
                    self.socket = None
                self.stats['reconnections'] += 1
            
            last_frame_time = current_time
            
            # Estadísticas cada 3 segundos
            if current_time - last_stats_time >= 3.0:
                self.print_stats()
                last_stats_time = current_time
    
    def print_stats(self):
        """Imprime estadísticas en tiempo real."""
        if self.stats['start_time'] is None:
            return
        
        elapsed = time.time() - self.stats['start_time']
        fps_actual = self.stats['frames_sent'] / elapsed if elapsed > 0 else 0
        success_rate = (self.stats['frames_sent'] / (self.stats['frames_sent'] + self.stats['frames_failed'])) * 100 if (self.stats['frames_sent'] + self.stats['frames_failed']) > 0 else 0
        mbps = (self.stats['bytes_sent'] / elapsed) / (1024**2) if elapsed > 0 else 0
        
        print(f"📊 [MJPEG] {self.stats['frames_sent']} frames, "
              f"{fps_actual:.1f} FPS, "
              f"{success_rate:.1f}% éxito, "
              f"{mbps:.2f} MB/s, "
              f"reconx: {self.stats['reconnections']}")
    
    def start_streaming(self):
        """Inicia el streaming MJPEG."""
        if self.running:
            print("⚠️ Streaming ya activo")
            return
        
        self.running = True
        self.stream_thread = threading.Thread(target=self.streaming_loop, daemon=True)
        self.stream_thread.start()
        print("🎬 Streaming MJPEG iniciado")
    
    def stop_streaming(self):
        """Detiene el streaming."""
        print("🛑 Deteniendo streaming MJPEG...")
        self.running = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=3.0)
        
        if self.socket:
            self.socket.close()
            self.socket = None
        
        self.camera.release()
        print("✅ Recursos liberados")
    
    def get_stats(self):
        """Retorna estadísticas detalladas."""
        if self.stats['start_time'] is None:
            return None
        
        elapsed = time.time() - self.stats['start_time']
        total_frames = self.stats['frames_sent'] + self.stats['frames_failed']
        
        return {
            'protocol': 'MJPEG_Stream',
            'frames_sent': self.stats['frames_sent'],
            'frames_failed': self.stats['frames_failed'],
            'total_frames': total_frames,
            'elapsed_time': elapsed,
            'fps_actual': self.stats['frames_sent'] / elapsed if elapsed > 0 else 0,
            'success_rate': (self.stats['frames_sent'] / total_frames) * 100 if total_frames > 0 else 0,
            'throughput_mbps': (self.stats['bytes_sent'] / elapsed) / (1024**2) if elapsed > 0 else 0,
            'total_mb': self.stats['bytes_sent'] / (1024**2),
            'connection_errors': self.stats['connection_errors'],
            'reconnections': self.stats['reconnections']
        }


def main():
    """Función principal del cliente MJPEG."""
    print("📺 MJPEG STREAM CLIENT - Streaming Continuo")
    print("Protocolo optimizado para baja latencia")
    print("=" * 50)
    
    # Crear cliente MJPEG
    client = MJPEGStreamClient()
    
    # Inicializar
    if not client.initialize():
        print("❌ Error en inicialización. Saliendo...")
        return False
    
    # Iniciar streaming
    client.start_streaming()
    
    try:
        print("🎬 Streaming MJPEG activo")
        print(f"📡 Servidor: {client.server_host}:{client.server_port}")
        print("📊 Estadísticas cada 3 segundos")
        print("⏹️ Presiona Ctrl+C para detener")
        print()
        
        # Mantener programa corriendo
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\n⚠️ Deteniendo por solicitud del usuario...")
        
    finally:
        client.stop_streaming()
        
        # Mostrar estadísticas finales
        stats = client.get_stats()
        if stats:
            print(f"\\n📊 RESUMEN FINAL - {stats['protocol']}:")
            print(f"   📤 Frames enviados: {stats['frames_sent']}")
            print(f"   ❌ Frames fallidos: {stats['frames_failed']}")
            print(f"   🎬 FPS promedio: {stats['fps_actual']:.1f}")
            print(f"   ✅ Tasa de éxito: {stats['success_rate']:.1f}%")
            print(f"   🚀 Throughput: {stats['throughput_mbps']:.2f} MB/s")
            print(f"   🔄 Reconexiones: {stats['reconnections']}")
            print(f"   📦 Total transmitido: {stats['total_mb']:.1f} MB")
    
    return True


if __name__ == "__main__":
    main()
