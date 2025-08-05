"""
Server - Receptor de Video Streaming

Este componente del servidor se encarga de:
- Recibir frames comprimidos desde el cliente
- Procesar/analizar el video (aquí irán los algoritmos de tracking)
- Mantener estadísticas de recepción
- Servir interfaz web para monitoreo

Ejecutar: python server_main.py
"""

import cv2
import numpy as np
import time
import base64
import threading
import sys
import os
from flask import Flask, request, jsonify, render_template, Response

# Agregar directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import *

class VideoReceiver:
    """
    Receptor de video streaming optimizado desde cliente remoto.
    
    Mejoras implementadas:
    - Métricas avanzadas de rendimiento
    - Detección de tipo de cliente (optimizado vs estándar)
    - Buffer más grande para mejor throughput
    - Medición de tiempo de procesamiento por frame
    """
    
    def __init__(self):
        """Inicializa el receptor de video optimizado."""
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Estadísticas de recepción mejoradas
        self.frames_received = 0
        self.bytes_received = 0
        self.start_time = None
        self.last_frame_time = 0
        self.processing_times = []  # Tiempos de procesamiento por frame
        
        # Detección de cliente optimizado
        self.client_type = "Unknown"  # TurboJPEG, OpenCV, etc.
        self.is_optimized_client = False
        
        # Buffer para procesar frames (aumentado para mejor rendimiento)
        self.frame_buffer = []
        self.max_buffer_size = 20  # Increased from 10 to 20 for better buffering
        
        print("VideoReceiver servidor optimizado inicializado")
        print(f"Configuración:")
        print(f"   Escuchando en puerto 5000")
        print(f"   Buffer máximo: {self.max_buffer_size} frames")
        print(f"   Métricas avanzadas: Habilitadas")
    
    def receive_frame(self, frame_data):
        """
        Recibe y procesa un frame desde el cliente optimizado.
        
        Args:
            frame_data (bytes): Frame JPEG comprimido
            
        Returns:
            bool: True si el frame se procesó correctamente
        """
        process_start_time = time.time()
        
        try:
            # Decodificar frame JPEG
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("❌ Error: No se pudo decodificar el frame")
                return False
            
            # Actualizar frame actual de forma thread-safe
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Agregar al buffer para procesamiento
            self.add_to_buffer(frame)
            
            # Calcular tiempo de procesamiento
            processing_time = (time.time() - process_start_time) * 1000
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:  # Mantener últimas 100 mediciones
                self.processing_times.pop(0)
            
            # Actualizar estadísticas
            self.frames_received += 1
            self.bytes_received += len(frame_data)
            self.last_frame_time = time.time()
            
            if self.start_time is None:
                self.start_time = time.time()
            
            # Detectar si es cliente optimizado basado en tamaño de frame
            frame_size_kb = len(frame_data) / 1024
            if frame_size_kb > 100 and frame.shape[0] >= 1080:  # Full HD
                if not self.is_optimized_client:
                    self.is_optimized_client = True
                    self.client_type = "Optimized"
                    print("Cliente optimizado detectado (capacidad Full HD)")
            
            # Log cada 50 frames
            if self.frames_received % 50 == 0:
                self.log_stats()
            
            return True
            
        except Exception as e:
            print(f"💥 Error procesando frame: {e}")
            return False
    
    def add_to_buffer(self, frame):
        """Agrega frame al buffer de procesamiento."""
        if len(self.frame_buffer) >= self.max_buffer_size:
            self.frame_buffer.pop(0)  # Remover frame más antiguo
        
        self.frame_buffer.append({
            'frame': frame,
            'timestamp': time.time()
        })
    
    def get_current_frame(self):
        """
        Obtiene el frame actual para visualización.
        
        Returns:
            numpy.ndarray: Frame actual o None
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def log_stats(self):
        """Imprime estadísticas de recepción optimizadas."""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        fps_received = self.frames_received / elapsed if elapsed > 0 else 0
        mbps = (self.bytes_received / elapsed) / (1024 * 1024) if elapsed > 0 else 0
        avg_frame_size = self.bytes_received / self.frames_received / 1024 if self.frames_received > 0 else 0
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        print(f"[SERVER-OPT] Stats: {self.frames_received} frames recibidos, "
              f"{fps_received:.1f} FPS, "
              f"{avg_frame_size:.1f} KB/frame, "
              f"{mbps:.2f} MB/s, "
              f"proc: {avg_processing_time:.1f}ms ({self.client_type})")
    
    def get_stats(self):
        """Retorna estadísticas detalladas optimizadas."""
        if self.start_time is None:
            return None
        
        elapsed = time.time() - self.start_time
        fps_received = self.frames_received / elapsed if elapsed > 0 else 0
        mbps = (self.bytes_received / elapsed) / (1024 * 1024) if elapsed > 0 else 0
        avg_frame_size = self.bytes_received / self.frames_received / 1024 if self.frames_received > 0 else 0
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'frames_received': self.frames_received,
            'elapsed_time': elapsed,
            'fps_received': fps_received,
            'mbps': mbps,
            'avg_frame_size_kb': avg_frame_size,
            'total_mb': self.bytes_received / (1024 * 1024),
            'buffer_size': len(self.frame_buffer),
            'last_frame_ago': time.time() - self.last_frame_time if self.last_frame_time > 0 else None,
            'avg_process_time_ms': avg_processing_time,
            'client_type': self.client_type,
            'is_optimized_client': self.is_optimized_client,
            'max_buffer_size': self.max_buffer_size
        }


# Instancia global del receptor
video_receiver = VideoReceiver()

# Configurar rutas para templates y archivos estáticos
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'static')

# Aplicación Flask para el servidor
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@app.route('/upload', methods=['POST'])
def upload_frame():
    """
    Endpoint para recibir frames desde el cliente.
    
    Espera datos en formato:
    - Content-Type: application/octet-stream
    - Body: Frame JPEG comprimido en bytes
    """
    try:
        # Obtener datos del frame
        frame_data = request.get_data()
        
        if not frame_data:
            return jsonify({'error': 'No frame data received'}), 400
        
        # Procesar frame
        success = video_receiver.receive_frame(frame_data)
        
        if success:
            return jsonify({
                'status': 'success',
                'frame_size': len(frame_data),
                'frames_total': video_receiver.frames_received
            })
        else:
            return jsonify({'error': 'Failed to process frame'}), 500
            
    except Exception as e:
        print(f"Error en upload endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Página principal con monitoreo del streaming."""
    stats = video_receiver.get_stats()
    has_frames = video_receiver.frames_received > 0
    is_active = has_frames and stats and stats['last_frame_ago'] < 10  # Activo si último frame hace menos de 10s
    
    return render_template('index.html', 
                         stats=stats,
                         has_frames=has_frames,
                         is_active=is_active)

@app.route('/current_frame')
def current_frame():
    """Endpoint para obtener el frame actual como imagen."""
    frame = video_receiver.get_current_frame()
    
    if frame is None:
        # Retornar imagen placeholder si no hay frame
        return "No frame available", 404
    
    # Convertir frame a JPEG
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    if not success:
        return "Error encoding frame", 500
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/stats')
def stats_json():
    """Endpoint para obtener estadísticas en JSON."""
    return jsonify(video_receiver.get_stats() or {})

def start_server():
    """Starts the optimized video streaming server."""
    print("Starting optimized video streaming server...")
    print(f"Configuration:")
    print(f"   Port: 5000")
    print(f"   Reception endpoint: /upload")
    print(f"   Web interface: http://localhost:5000")
    print(f"   Stats API: http://localhost:5000/stats")
    print(f"   Optimizations: TurboJPEG + Multithreading support")
    print(f"   Enhanced buffer: 20 frames")
    print()
    print("Server optimized for clients with:")
    print("  TurboJPEG compression")
    print("  Multithreaded architecture")
    print("  Advanced performance metrics")
    print()
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    finally:
        print("Closing optimized server...")

if __name__ == "__main__":
    start_server()
