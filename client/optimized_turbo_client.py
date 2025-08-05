"""
High Performance Client - Cliente Optimizado con TurboJPEG y Multithreading

Implementa optimizaciones avanzadas para maximizar FPS en transmisión Full HD:
- TurboJPEG para compresión 3-5x más rápida que OpenCV
- Sistema multithreaded con colas producer-consumer
- Captura, codificación y envío desacoplados
- Buffer circular para manejo eficiente de frames

Ejecutar: python optimized_turbo_client.py
"""

import cv2
import time
import threading
import queue
import sys
import os
import requests
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import deque
import psutil

# Intentar importar TurboJPEG (requiere instalación: pip install PyTurboJPEG)
try:
    from turbojpeg import TurboJPEG
    TURBOJPEG_AVAILABLE = True
    print("✅ TurboJPEG disponible - Compresión optimizada habilitada")
except ImportError:
    TURBOJPEG_AVAILABLE = False
    print("⚠️ TurboJPEG no disponible - Usando OpenCV estándar")
    print("💡 Para instalar: pip install PyTurboJPEG")

# Agregar directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.camera_handler import CameraHandler
from shared.config import *

class HighPerformanceStreamer:
    """
    Cliente de streaming de alto rendimiento con optimizaciones avanzadas.
    
    Características:
    - TurboJPEG para compresión ultra-rápida
    - Arquitectura multithreaded producer-consumer
    - Buffers circulares para evitar bloqueos
    - Métricas detalladas de rendimiento
    """
    
    def __init__(self, server_url=None, target_fps=None, jpeg_quality=None, buffer_size=30):
        """
        Inicializa el cliente de alto rendimiento.
        
        Args:
            server_url: URL del servidor
            target_fps: FPS objetivo
            jpeg_quality: Calidad JPEG (1-100)
            buffer_size: Tamaño del buffer circular
        """
        self.server_url = server_url or SERVER_URL
        self.target_fps = target_fps or STREAMING_FPS
        self.jpeg_quality = jpeg_quality or JPEG_QUALITY
        self.frame_interval = 1.0 / self.target_fps
        self.buffer_size = buffer_size
        
        # Inicializar TurboJPEG si está disponible
        if TURBOJPEG_AVAILABLE:
            self.turbo_jpeg = TurboJPEG()
            self.compression_method = "TurboJPEG"
        else:
            self.turbo_jpeg = None
            self.compression_method = "OpenCV"
        
        # Componentes
        self.camera = CameraHandler()
        
        # Colas para comunicación entre threads
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.compressed_queue = queue.Queue(maxsize=buffer_size)
        
        # Control de threads
        self.running = False
        self.capture_thread = None
        self.compression_threads = []
        self.sender_thread = None
        self.stats_thread = None
        
        # Estadísticas detalladas
        self.stats = {
            'frames_captured': 0,
            'frames_compressed': 0,
            'frames_sent': 0,
            'frames_failed': 0,
            'frames_dropped_capture': 0,
            'frames_dropped_compression': 0,
            'bytes_sent': 0,
            'compression_times': deque(maxlen=1000),
            'send_times': deque(maxlen=1000),
            'capture_times': deque(maxlen=1000),
            'start_time': None
        }
        
        # Configurar sesión HTTP optimizada
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=20, pool_block=False)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        print("🚀 HighPerformanceStreamer inicializado")
        print(f"📊 Configuración:")
        print(f"   Servidor: {self.server_url}")
        print(f"   Target FPS: {self.target_fps}")
        print(f"   Calidad JPEG: {self.jpeg_quality}%")
        print(f"   Método compresión: {self.compression_method}")
        print(f"   Buffer size: {self.buffer_size}")
    
    def initialize(self):
        """Inicializa el sistema de streaming."""
        print("🔧 Inicializando sistema de alto rendimiento...")
        
        # Inicializar cámara
        if not self.camera.initialize():
            print("❌ Error inicializando cámara")
            return False
        
        # Verificar conexión al servidor
        if not self.test_server_connection():
            print("❌ Error conectando al servidor")
            return False
        
        print("✅ Inicialización exitosa")
        return True
    
    def test_server_connection(self):
        """Verifica conexión al servidor."""
        try:
            test_url = self.server_url.replace('/upload', '/stats')
            response = self.session.get(test_url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ Servidor conectado: {test_url}")
                return True
            else:
                print(f"❌ Servidor respondió con código: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error conectando: {e}")
            return False
    
    def compress_frame_turbojpeg(self, frame):
        """Comprime frame usando TurboJPEG (más rápido)."""
        start_time = time.time()
        try:
            # TurboJPEG espera BGR (OpenCV format)
            compressed_data = self.turbo_jpeg.encode(frame, quality=self.jpeg_quality)
            compression_time = (time.time() - start_time) * 1000
            self.stats['compression_times'].append(compression_time)
            return compressed_data
        except Exception as e:
            print(f"❌ Error TurboJPEG: {e}")
            return None
    
    def compress_frame_opencv(self, frame):
        """Comprime frame usando OpenCV estándar."""
        start_time = time.time()
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, encoded_frame = cv2.imencode('.jpg', frame, encode_params)
        compression_time = (time.time() - start_time) * 1000
        self.stats['compression_times'].append(compression_time)
        
        if not success:
            return None
        return encoded_frame.tobytes()
    
    def compress_frame(self, frame):
        """Comprime frame usando el método disponible."""
        if self.turbo_jpeg:
            return self.compress_frame_turbojpeg(frame)
        else:
            return self.compress_frame_opencv(frame)
    
    def capture_thread_worker(self):
        """Worker thread para captura de frames."""
        print("📷 Thread de captura iniciado")
        last_capture_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Control de tasa de captura
            if current_time - last_capture_time < self.frame_interval:
                time.sleep(0.001)
                continue
            
            start_time = time.time()
            ret, frame = self.camera.get_frame()
            capture_time = (time.time() - start_time) * 1000
            
            if ret and frame is not None:
                self.stats['capture_times'].append(capture_time)
                
                try:
                    # Intentar poner frame en cola (no bloqueante)
                    self.frame_queue.put_nowait((frame.copy(), current_time))
                    self.stats['frames_captured'] += 1
                except queue.Full:
                    # Buffer lleno, frame descartado
                    self.stats['frames_dropped_capture'] += 1
            
            last_capture_time = current_time
        
        print("📷 Thread de captura finalizado")
    
    def compression_thread_worker(self):
        """Worker thread para compresión de frames."""
        print("🗜️ Thread de compresión iniciado")
        
        while self.running:
            try:
                # Obtener frame de la cola
                frame, timestamp = self.frame_queue.get(timeout=1.0)
                
                # Comprimir frame
                compressed_data = self.compress_frame(frame)
                
                if compressed_data is not None:
                    try:
                        # Poner frame comprimido en cola de envío
                        self.compressed_queue.put_nowait((compressed_data, timestamp))
                        self.stats['frames_compressed'] += 1
                    except queue.Full:
                        self.stats['frames_dropped_compression'] += 1
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error en compresión: {e}")
        
        print("🗜️ Thread de compresión finalizado")
    
    def sender_thread_worker(self):
        """Worker thread para envío de frames."""
        print("📤 Thread de envío iniciado")
        
        while self.running:
            try:
                # Obtener frame comprimido
                compressed_data, timestamp = self.compressed_queue.get(timeout=1.0)
                
                # Enviar frame
                start_time = time.time()
                success = self.send_frame(compressed_data)
                send_time = (time.time() - start_time) * 1000
                
                self.stats['send_times'].append(send_time)
                
                if success:
                    self.stats['frames_sent'] += 1
                    self.stats['bytes_sent'] += len(compressed_data)
                else:
                    self.stats['frames_failed'] += 1
                
                self.compressed_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error en envío: {e}")
        
        print("📤 Thread de envío finalizado")
    
    def send_frame(self, frame_data):
        """Envía frame al servidor."""
        try:
            headers = {
                'Content-Type': 'application/octet-stream',
                'Content-Length': str(len(frame_data))
            }
            
            response = self.session.post(
                self.server_url,
                data=frame_data,
                headers=headers,
                timeout=1.0  # Timeout muy corto para alta velocidad
            )
            
            return response.status_code == 200
            
        except Exception:
            return False
    
    def stats_thread_worker(self):
        """Worker thread para estadísticas en tiempo real."""
        while self.running:
            time.sleep(2.0)  # Estadísticas cada 2 segundos
            self.print_realtime_stats()
    
    def print_realtime_stats(self):
        """Imprime estadísticas en tiempo real."""
        if self.stats['start_time'] is None:
            return
        
        elapsed = time.time() - self.stats['start_time']
        
        # FPS reales
        capture_fps = self.stats['frames_captured'] / elapsed if elapsed > 0 else 0
        compression_fps = self.stats['frames_compressed'] / elapsed if elapsed > 0 else 0
        send_fps = self.stats['frames_sent'] / elapsed if elapsed > 0 else 0
        
        # Tiempos promedio
        avg_capture = np.mean(self.stats['capture_times']) if self.stats['capture_times'] else 0
        avg_compression = np.mean(self.stats['compression_times']) if self.stats['compression_times'] else 0
        avg_send = np.mean(self.stats['send_times']) if self.stats['send_times'] else 0
        
        # Tamaños de cola
        frame_queue_size = self.frame_queue.qsize()
        compressed_queue_size = self.compressed_queue.qsize()
        
        # Throughput
        mbps = (self.stats['bytes_sent'] / elapsed) / (1024**2) if elapsed > 0 else 0
        
        print(f"📊 [STATS] Cap:{capture_fps:.1f} Comp:{compression_fps:.1f} Send:{send_fps:.1f} FPS | "
              f"Q:{frame_queue_size}/{compressed_queue_size} | "
              f"T:{avg_capture:.1f}/{avg_compression:.1f}/{avg_send:.1f}ms | "
              f"{mbps:.2f}MB/s")
    
    def start_streaming(self):
        """Inicia el streaming multithreaded."""
        if self.running:
            print("⚠️ Streaming ya activo")
            return
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Iniciar threads
        self.capture_thread = threading.Thread(target=self.capture_thread_worker, daemon=True)
        self.capture_thread.start()
        
        # Múltiples threads de compresión para aprovechar CPU multicores
        num_compression_threads = min(4, psutil.cpu_count())
        for i in range(num_compression_threads):
            thread = threading.Thread(target=self.compression_thread_worker, daemon=True)
            thread.start()
            self.compression_threads.append(thread)
        
        self.sender_thread = threading.Thread(target=self.sender_thread_worker, daemon=True)
        self.sender_thread.start()
        
        self.stats_thread = threading.Thread(target=self.stats_thread_worker, daemon=True)
        self.stats_thread.start()
        
        print(f"🚀 Streaming multithreaded iniciado ({num_compression_threads} compression threads)")
    
    def stop_streaming(self):
        """Detiene el streaming y limpia recursos."""
        print("🛑 Deteniendo streaming...")
        self.running = False
        
        # Esperar que terminen los threads
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        for thread in self.compression_threads:
            thread.join(timeout=2.0)
        
        if self.sender_thread:
            self.sender_thread.join(timeout=2.0)
        
        if self.stats_thread:
            self.stats_thread.join(timeout=2.0)
        
        # Limpiar recursos
        self.camera.release()
        self.session.close()
        
        print("✅ Recursos liberados")
    
    def get_performance_summary(self):
        """Retorna resumen de rendimiento."""
        if self.stats['start_time'] is None:
            return {}
        
        elapsed = time.time() - self.stats['start_time']
        
        return {
            'method': self.compression_method,
            'elapsed_time': elapsed,
            'frames_captured': self.stats['frames_captured'],
            'frames_sent': self.stats['frames_sent'],
            'frames_failed': self.stats['frames_failed'],
            'frames_dropped': self.stats['frames_dropped_capture'] + self.stats['frames_dropped_compression'],
            'fps_capture': self.stats['frames_captured'] / elapsed if elapsed > 0 else 0,
            'fps_sent': self.stats['frames_sent'] / elapsed if elapsed > 0 else 0,
            'success_rate': (self.stats['frames_sent'] / (self.stats['frames_sent'] + self.stats['frames_failed'])) * 100 if (self.stats['frames_sent'] + self.stats['frames_failed']) > 0 else 0,
            'avg_compression_ms': np.mean(self.stats['compression_times']) if self.stats['compression_times'] else 0,
            'avg_send_ms': np.mean(self.stats['send_times']) if self.stats['send_times'] else 0,
            'throughput_mbps': (self.stats['bytes_sent'] / elapsed) / (1024**2) if elapsed > 0 else 0,
            'total_mb': self.stats['bytes_sent'] / (1024**2)
        }


def main():
    """Función principal del cliente optimizado."""
    print("🚀 HIGH PERFORMANCE CLIENT - TurboJPEG + Multithreading")
    print("Optimizado para máximo rendimiento Full HD")
    print("=" * 60)
    
    # Crear cliente optimizado
    client = HighPerformanceStreamer()
    
    # Inicializar
    if not client.initialize():
        print("❌ Error en inicialización. Saliendo...")
        return False
    
    # Iniciar streaming
    client.start_streaming()
    
    try:
        print("🔥 Streaming de alto rendimiento activo")
        print(f"📡 Servidor: {client.server_url}")
        print(f"🗜️ Método: {client.compression_method}")
        print("📊 Estadísticas en tiempo real cada 2 segundos")
        print("⏹️ Presiona Ctrl+C para detener")
        print()
        
        # Mantener programa corriendo
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Deteniendo por solicitud del usuario...")
        
    finally:
        client.stop_streaming()
        
        # Mostrar resumen final
        summary = client.get_performance_summary()
        if summary:
            print(f"\n📊 RESUMEN FINAL - {summary['method']}:")
            print(f"   🎬 FPS capturados: {summary['fps_capture']:.1f}")
            print(f"   📤 FPS enviados: {summary['fps_sent']:.1f}")
            print(f"   ✅ Tasa de éxito: {summary['success_rate']:.1f}%")
            print(f"   🗜️ Compresión promedio: {summary['avg_compression_ms']:.1f}ms")
            print(f"   📡 Envío promedio: {summary['avg_send_ms']:.1f}ms")
            print(f"   🚀 Throughput: {summary['throughput_mbps']:.2f} MB/s")
            print(f"   🗂️ Total transmitido: {summary['total_mb']:.1f} MB")
    
    return True


if __name__ == "__main__":
    main()
