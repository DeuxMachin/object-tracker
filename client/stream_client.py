"""
Stream Client - Transmisión de Video Local a Remoto

Este componente se encarga de:
- Capturar video local usando camera_handler.py
- Comprimir frames a JPEG para transmisión eficiente
- Enviar flujo de datos al servidor remoto
- Mantener tasa de transmisión estable y tolerante a errores

Pruebas locales: Crea servidor web Flask para verificar transmisión
Objetivo: ~10 FPS, ~100KB por frame, latencia <10ms
"""

import cv2
import time
import threading
import sys
import os
from flask import Flask, Response, render_template_string

# Agregar directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_handler import CameraHandler

class VideoStreamer:
    """
    Manejador de transmisión de video en tiempo real.
    
    Comprime y transmite frames desde cámara local hacia servidor remoto,
    manteniendo control de tasa de frames y calidad de compresión.
    """
    
    def __init__(self, target_fps=10, jpeg_quality=80):
        """
        Inicializa el streamer de video.
        
        Args:
            target_fps: Frames por segundo objetivo (default: 10)
            jpeg_quality: Calidad JPEG 1-100 (default: 80)
        """
        self.camera = CameraHandler()
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.frame_interval = 1.0 / target_fps  # Intervalo entre frames
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Estadísticas de rendimiento
        self.frames_sent = 0
        self.bytes_sent = 0
        self.start_time = None
        
    def initialize(self):
        """Inicializa la cámara y prepara el streaming."""
        print("Inicializando sistema de streaming...")
        if not self.camera.initialize():
            print("Error: No se pudo inicializar la cámara")
            return False
        
        print(f"Configuración de streaming:")
        print(f"   Target FPS: {self.target_fps}")
        print(f"   Calidad JPEG: {self.jpeg_quality}%")
        print(f"   Intervalo entre frames: {self.frame_interval:.3f}s")
        return True
    
    def compress_frame(self, frame):
        """
        Comprime un frame a formato JPEG.
        
        Args:
            frame: Frame de video como array numpy
            
        Returns:
            bytes: Frame comprimido en JPEG
        """
        # Parámetros de compresión JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        
        # Comprimir frame
        success, encoded_frame = cv2.imencode('.jpg', frame, encode_params)
        
        if not success:
            return None
            
        return encoded_frame.tobytes()
    
    def capture_loop(self):
        """
        Loop principal de captura y compresión de frames.
        Mantiene la tasa de FPS objetivo.
        """
        print("Iniciando captura de video...")
        self.start_time = time.time()
        last_frame_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Control de tasa de frames
            if current_time - last_frame_time < self.frame_interval:
                time.sleep(0.001)  # Sleep corto para no saturar CPU
                continue
            
            # Capturar frame
            ret, frame = self.camera.get_frame()
            
            if not ret or frame is None:
                print("Frame inválido, continuando...")
                continue
            
            # Comprimir frame
            compressed_frame = self.compress_frame(frame)
            
            if compressed_frame is None:
                print("Error comprimiendo frame")
                continue
            
            # Actualizar frame actual de forma thread-safe
            with self.frame_lock:
                self.current_frame = compressed_frame
            
            # Actualizar estadísticas
            self.frames_sent += 1
            self.bytes_sent += len(compressed_frame)
            last_frame_time = current_time
            
            # Log de rendimiento cada 100 frames
            if self.frames_sent % 100 == 0:
                elapsed = current_time - self.start_time
                fps_actual = self.frames_sent / elapsed
                mbps = (self.bytes_sent / elapsed) / (1024 * 1024)
                avg_frame_size = self.bytes_sent / self.frames_sent / 1024
                
                print(f"Stats: {self.frames_sent} frames, "
                      f"{fps_actual:.1f} FPS real, "
                      f"{avg_frame_size:.1f} KB/frame, "
                      f"{mbps:.2f} MB/s")
    
    def get_frame(self):
        """
        Obtiene el frame actual comprimido.
        
        Returns:
            bytes: Frame JPEG comprimido o None
        """
        with self.frame_lock:
            return self.current_frame
    
    def start_streaming(self):
        """Inicia el streaming en un hilo separado."""
        if self.running:
            print("Streaming ya está activo")
            return
        
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print("Streaming iniciado")
    
    def stop_streaming(self):
        """Detiene el streaming y libera recursos."""
        print("Deteniendo streaming...")
        self.running = False
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        
        self.camera.release()
        print("Recursos liberados")
    
    def get_stats(self):
        """Retorna estadísticas actuales de streaming."""
        if self.start_time is None:
            return None
        
        elapsed = time.time() - self.start_time
        fps_actual = self.frames_sent / elapsed if elapsed > 0 else 0
        mbps = (self.bytes_sent / elapsed) / (1024 * 1024) if elapsed > 0 else 0
        avg_frame_size = self.bytes_sent / self.frames_sent / 1024 if self.frames_sent > 0 else 0
        
        return {
            'frames_sent': self.frames_sent,
            'elapsed_time': elapsed,
            'fps_actual': fps_actual,
            'mbps': mbps,
            'avg_frame_size_kb': avg_frame_size,
            'total_mb': self.bytes_sent / (1024 * 1024)
        }


# Instancia global del streamer
video_streamer = VideoStreamer()

# Aplicación Flask para pruebas locales
app = Flask(__name__)

def generate_frames():
    """
    Generador de frames para Flask streaming.
    
    Yields:
        bytes: Frame JPEG en formato multipart para streaming HTTP
    """
    while True:
        frame = video_streamer.get_frame()
        
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Formato multipart para streaming HTTP
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Página principal con el video stream."""
    template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Streaming Test</title>
        <style>
            body { font-family: Arial; text-align: center; background: #f0f0f0; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            .video-container { margin: 20px 0; }
            img { max-width: 100%; border: 2px solid #333; }
            .stats { text-align: left; background: #fff; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Video Streaming Test</h1>
            <p>Transmisión desde cámara local con compresión JPEG</p>
            
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
            
            <div class="stats">
                <h3>Información del Sistema:</h3>
                <p><strong>Target FPS:</strong> {{ target_fps }}</p>
                <p><strong>Calidad JPEG:</strong> {{ jpeg_quality }}%</p>
                <p><strong>Estado:</strong> <span style="color: green;">Activo</span></p>
                <p><em>Presiona Ctrl+C en la consola para detener</em></p>
            </div>
        </div>
        
        <script>
            // Auto-refresh de la página cada 30 segundos para mostrar stats actualizadas
            setTimeout(function(){ location.reload(); }, 30000);
        </script>
    </body>
    </html>
    '''
    
    return render_template_string(template, 
                                target_fps=video_streamer.target_fps,
                                jpeg_quality=video_streamer.jpeg_quality)

@app.route('/video_feed')
def video_feed():
    """Endpoint de streaming de video."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Endpoint para obtener estadísticas en JSON."""
    return video_streamer.get_stats() or {}

def test_local_streaming():
    """
    Función de prueba para streaming local.
    
    Inicia el servidor Flask y el streaming de video para verificar
    que la compresión y transmisión funcionan correctamente.
    """
    print("Iniciando test de streaming local...")
    
    # Inicializar streamer
    if not video_streamer.initialize():
        return False
    
    # Iniciar streaming
    video_streamer.start_streaming()
    
    try:
        print("Iniciando servidor web local...")
        print("Abre tu navegador en: http://localhost:5000")
        print("Deberías ver el video de tu cámara en tiempo real")
        print("Presiona Ctrl+C para detener")
        
        # Ejecutar servidor Flask
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nDeteniendo por solicitud del usuario...")
        
    finally:
        video_streamer.stop_streaming()
        
        # Mostrar estadísticas finales
        stats = video_streamer.get_stats()
        if stats:
            print("\nEstadísticas finales:")
            print(f"   Frames enviados: {stats['frames_sent']}")
            print(f"   Tiempo total: {stats['elapsed_time']:.1f}s")
            print(f"   FPS promedio: {stats['fps_actual']:.1f}")
            print(f"   Tamaño promedio: {stats['avg_frame_size_kb']:.1f} KB/frame")
            print(f"   Throughput: {stats['mbps']:.2f} MB/s")
            print(f"   Total transmitido: {stats['total_mb']:.1f} MB")
    
    return True

if __name__ == "__main__":
    test_local_streaming()
