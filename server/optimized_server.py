"""
Optimized Server - Servidor con Soporte para Múltiples Protocolos

Soporta:
- HTTP POST tradicional (compatibilidad con cliente original)
- MJPEG Stream continuo (para cliente optimizado)
- Métricas avanzadas de rendimiento
- Interfaz web mejorada

Ejecutar: python optimized_server.py
"""

import cv2
import numpy as np
import time
import threading
import sys
import os
import socket
import select
from flask import Flask, request, jsonify, render_template, Response
from collections import deque

# Agregar directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import *

class OptimizedVideoReceiver:
    """
    Receptor de video optimizado con soporte para múltiples protocolos.
    
    Soporta:
    - HTTP POST (cliente original)
    - MJPEG Stream (cliente optimizado)
    """
    
    def __init__(self):
        """Inicializa el receptor optimizado."""
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Estadísticas detalladas
        self.stats = {
            'http_frames': 0,
            'mjpeg_frames': 0,
            'total_bytes': 0,
            'start_time': None,
            'last_frame_time': 0,
            'frame_times': deque(maxlen=1000),
            'frame_sizes': deque(maxlen=1000),
            'protocol_used': 'None'
        }
        
        # Buffer para frames
        self.frame_buffer = deque(maxlen=20)
        
        # MJPEG Server
        self.mjpeg_running = False
        self.mjpeg_thread = None
        self.mjpeg_socket = None
        
        print("🚀 OptimizedVideoReceiver inicializado")
        print(f"📊 Configuración:")
        print(f"   HTTP Port: 5000")
        print(f"   MJPEG Port: 5001")
        print(f"   Buffer máximo: 20 frames")
    
    def receive_frame(self, frame_data, protocol='HTTP'):
        """
        Recibe y procesa frame desde cualquier protocolo.
        
        Args:
            frame_data: Frame JPEG comprimido
            protocol: 'HTTP' o 'MJPEG'
        """
        try:
            start_time = time.time()
            
            # Decodificar frame
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return False
            
            # Actualizar frame actual thread-safe
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Agregar al buffer
            self.frame_buffer.append({
                'frame': frame,
                'timestamp': time.time(),
                'size': len(frame_data),
                'protocol': protocol
            })
            
            # Actualizar estadísticas
            if protocol == 'HTTP':
                self.stats['http_frames'] += 1
            else:
                self.stats['mjpeg_frames'] += 1
            
            self.stats['total_bytes'] += len(frame_data)
            self.stats['last_frame_time'] = time.time()
            self.stats['protocol_used'] = protocol
            
            process_time = (time.time() - start_time) * 1000
            self.stats['frame_times'].append(process_time)
            self.stats['frame_sizes'].append(len(frame_data) / 1024)
            
            if self.stats['start_time'] is None:
                self.stats['start_time'] = time.time()
            
            # Log cada 100 frames
            total_frames = self.stats['http_frames'] + self.stats['mjpeg_frames']
            if total_frames % 100 == 0:
                self.log_stats()
            
            return True
            
        except Exception as e:
            print(f"❌ Error procesando frame: {e}")
            return False
    
    def get_current_frame(self):
        """Obtiene frame actual para visualización."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def log_stats(self):
        """Imprime estadísticas de recepción."""
        if self.stats['start_time'] is None:
            return
        
        elapsed = time.time() - self.stats['start_time']
        total_frames = self.stats['http_frames'] + self.stats['mjpeg_frames']
        fps = total_frames / elapsed if elapsed > 0 else 0
        mbps = (self.stats['total_bytes'] / elapsed) / (1024**2) if elapsed > 0 else 0
        avg_size = np.mean(self.stats['frame_sizes']) if self.stats['frame_sizes'] else 0
        avg_process_time = np.mean(self.stats['frame_times']) if self.stats['frame_times'] else 0
        
        print(f"[SERVER] {total_frames} frames ({self.stats['http_frames']} HTTP, {self.stats['mjpeg_frames']} MJPEG), "
              f"{fps:.1f} FPS, {avg_size:.1f} KB/frame, {mbps:.2f} MB/s, "
              f"{avg_process_time:.1f}ms process")
    
    def get_detailed_stats(self):
        """Retorna estadísticas detalladas."""
        if self.stats['start_time'] is None:
            return None
        
        elapsed = time.time() - self.stats['start_time']
        total_frames = self.stats['http_frames'] + self.stats['mjpeg_frames']
        
        return {
            'total_frames': total_frames,
            'http_frames': self.stats['http_frames'],
            'mjpeg_frames': self.stats['mjpeg_frames'],
            'elapsed_time': elapsed,
            'fps_total': total_frames / elapsed if elapsed > 0 else 0,
            'fps_http': self.stats['http_frames'] / elapsed if elapsed > 0 else 0,
            'fps_mjpeg': self.stats['mjpeg_frames'] / elapsed if elapsed > 0 else 0,
            'mbps': (self.stats['total_bytes'] / elapsed) / (1024**2) if elapsed > 0 else 0,
            'total_mb': self.stats['total_bytes'] / (1024**2),
            'avg_frame_size_kb': np.mean(self.stats['frame_sizes']) if self.stats['frame_sizes'] else 0,
            'avg_process_time_ms': np.mean(self.stats['frame_times']) if self.stats['frame_times'] else 0,
            'buffer_size': len(self.frame_buffer),
            'last_frame_ago': time.time() - self.stats['last_frame_time'] if self.stats['last_frame_time'] > 0 else None,
            'protocol_used': self.stats['protocol_used']
        }
    
    def start_mjpeg_server(self, port=5001):
        """Inicia servidor MJPEG en puerto separado."""
        def mjpeg_server():
            try:
                self.mjpeg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.mjpeg_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.mjpeg_socket.bind(('0.0.0.0', port))
                self.mjpeg_socket.listen(5)
                print(f"📺 Servidor MJPEG iniciado en puerto {port}")
                
                while self.mjpeg_running:
                    try:
                        # Usar select para timeout
                        ready, _, _ = select.select([self.mjpeg_socket], [], [], 1.0)
                        if not ready:
                            continue
                        
                        client_socket, addr = self.mjpeg_socket.accept()
                        print(f"📺 Cliente MJPEG conectado desde {addr}")
                        
                        # Manejar cliente MJPEG en thread separado
                        client_thread = threading.Thread(
                            target=self.handle_mjpeg_client, 
                            args=(client_socket,), 
                            daemon=True
                        )
                        client_thread.start()
                        
                    except socket.error:
                        if self.mjpeg_running:
                            print("❌ Error en servidor MJPEG")
                        break
                        
            except Exception as e:
                print(f"❌ Error iniciando servidor MJPEG: {e}")
            finally:
                if self.mjpeg_socket:
                    self.mjpeg_socket.close()
        
        self.mjpeg_running = True
        self.mjpeg_thread = threading.Thread(target=mjpeg_server, daemon=True)
        self.mjpeg_thread.start()
    
    def handle_mjpeg_client(self, client_socket):
        """Maneja cliente MJPEG individual."""
        try:
            # Leer headers HTTP
            headers = b""
            while b"\\r\\n\\r\\n" not in headers:
                data = client_socket.recv(1024)
                if not data:
                    break
                headers += data
            
            print("📺 Headers MJPEG recibidos, iniciando recepción de frames...")
            
            # Procesar stream MJPEG
            buffer = b""
            while self.mjpeg_running:
                try:
                    data = client_socket.recv(8192)
                    if not data:
                        break
                    
                    buffer += data
                    
                    # Buscar boundary frames
                    while b"--mjpegboundary" in buffer:
                        # Encontrar inicio del frame
                        boundary_start = buffer.find(b"--mjpegboundary")
                        if boundary_start == -1:
                            break
                        
                        # Buscar Content-Length
                        header_end = buffer.find(b"\\r\\n\\r\\n", boundary_start)
                        if header_end == -1:
                            break
                        
                        headers_section = buffer[boundary_start:header_end].decode('utf-8', errors='ignore')
                        
                        # Extraer Content-Length
                        content_length = 0
                        for line in headers_section.split('\\n'):
                            if 'Content-Length:' in line:
                                content_length = int(line.split(':')[1].strip())
                                break
                        
                        frame_start = header_end + 4
                        frame_end = frame_start + content_length
                        
                        # Verificar que tenemos el frame completo
                        if len(buffer) >= frame_end:
                            frame_data = buffer[frame_start:frame_end]
                            
                            # Procesar frame
                            self.receive_frame(frame_data, 'MJPEG')
                            
                            # Remover frame procesado del buffer
                            buffer = buffer[frame_end:]
                        else:
                            break
                
                except socket.error:
                    break
                except Exception as e:
                    print(f"❌ Error procesando MJPEG: {e}")
                    break
        
        except Exception as e:
            print(f"❌ Error en cliente MJPEG: {e}")
        finally:
            client_socket.close()
            print("📺 Cliente MJPEG desconectado")
    
    def stop_mjpeg_server(self):
        """Detiene servidor MJPEG."""
        self.mjpeg_running = False
        if self.mjpeg_socket:
            self.mjpeg_socket.close()
        if self.mjpeg_thread:
            self.mjpeg_thread.join(timeout=2.0)


# Instancia global del receptor
video_receiver = OptimizedVideoReceiver()

# Configurar Flask
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@app.route('/upload', methods=['POST'])
def upload_frame():
    """Endpoint HTTP tradicional para recibir frames."""
    try:
        frame_data = request.get_data()
        
        if not frame_data:
            return jsonify({'error': 'No frame data received'}), 400
        
        success = video_receiver.receive_frame(frame_data, 'HTTP')
        
        if success:
            return jsonify({
                'status': 'success',
                'frame_size': len(frame_data),
                'protocol': 'HTTP'
            })
        else:
            return jsonify({'error': 'Failed to process frame'}), 500
            
    except Exception as e:
        print(f"❌ Error en upload HTTP: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Página principal con monitoreo mejorado."""
    stats = video_receiver.get_detailed_stats()
    has_frames = stats and stats['total_frames'] > 0
    is_active = has_frames and stats and stats['last_frame_ago'] < 10
    
    return render_template('optimized_index.html', 
                         stats=stats,
                         has_frames=has_frames,
                         is_active=is_active)

@app.route('/current_frame')
def current_frame():
    """Endpoint para frame actual."""
    frame = video_receiver.get_current_frame()
    
    if frame is None:
        return "No frame available", 404
    
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    if not success:
        return "Error encoding frame", 500
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/stats')
def stats_json():
    """API de estadísticas detalladas."""
    return jsonify(video_receiver.get_detailed_stats() or {})

def start_optimized_server():
    """Inicia el servidor optimizado con ambos protocolos."""
    print("🚀 INICIANDO SERVIDOR OPTIMIZADO")
    print("=" * 50)
    print(f"📊 Configuración:")
    print(f"   HTTP Server: Puerto 5000")
    print(f"   MJPEG Server: Puerto 5001")
    print(f"   Interfaz web: http://localhost:5000")
    print(f"   API Stats: http://localhost:5000/stats")
    print()
    print("🔌 Protocolos soportados:")
    print("   • HTTP POST (cliente original)")
    print("   • MJPEG Stream (cliente optimizado)")
    print()
    
    # Iniciar servidor MJPEG
    video_receiver.start_mjpeg_server(5001)
    
    print("⚡ Servidores iniciados. Presiona Ctrl+C para detener")
    
    try:
        # Iniciar Flask
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\\n⚠️ Servidor detenido por el usuario")
    finally:
        print("🛑 Cerrando servidores...")
        video_receiver.stop_mjpeg_server()
        print("✅ Servidores cerrados")

if __name__ == "__main__":
    start_optimized_server()
