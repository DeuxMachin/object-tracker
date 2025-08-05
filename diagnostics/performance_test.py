"""
Performance Test - Diagnóstico de Rendimiento para Video Streaming

Este módulo evalúa la capacidad del sistema cliente-servidor para transmitir
video en tiempo real a diferentes resoluciones, especialmente 1920×1080.

Métricas evaluadas:
- FPS real vs objetivo
- Tamaño promedio de frame
- Throughput de red (MB/s)
- Porcentaje de éxito de transmisión
- Latencia de ida y vuelta
- Uso de CPU/memoria
- Pérdida de frames

Ejecutar: python performance_test.py
"""

import cv2
import time
import threading
import sys
import os
import requests
import numpy as np
import psutil
import json
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Agregar directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.camera_handler import CameraHandler
from shared.config import *

@dataclass
class TestConfiguration:
    """Configuración para una prueba de rendimiento."""
    name: str
    width: int
    height: int
    fps: int
    jpeg_quality: int
    duration_seconds: int = 60  # Duración de la prueba
    warmup_seconds: int = 10    # Tiempo de calentamiento

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento de una prueba."""
    config_name: str
    
    # Métricas de transmisión
    frames_sent: int = 0
    frames_failed: int = 0
    bytes_sent: int = 0
    test_duration: float = 0.0
    
    # FPS y timing
    fps_actual: float = 0.0
    fps_target: float = 0.0
    frame_times: List[float] = None
    
    # Tamaños y throughput
    avg_frame_size_kb: float = 0.0
    min_frame_size_kb: float = 0.0
    max_frame_size_kb: float = 0.0
    throughput_mbps: float = 0.0
    
    # Latencia
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Calidad y eficiencia
    success_rate: float = 0.0
    compression_ratio: float = 0.0
    
    # Recursos del sistema
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    
    def __post_init__(self):
        if self.frame_times is None:
            self.frame_times = []

class PerformanceTester:
    """
    Probador de rendimiento para streaming de video.
    
    Ejecuta pruebas exhaustivas con diferentes configuraciones
    y recopila métricas detalladas de rendimiento.
    """
    
    def __init__(self, server_url: str = None):
        """
        Inicializa el probador de rendimiento.
        
        Args:
            server_url: URL del servidor (default: desde config)
        """
        self.server_url = server_url or SERVER_URL
        self.camera = None
        
        # Configuraciones de prueba predefinidas
        self.test_configs = [
            TestConfiguration("Low_Quality", 640, 480, 15, 60),
            TestConfiguration("Medium_Quality", 1280, 720, 20, 70),
            TestConfiguration("High_Quality", 1920, 1080, 25, 75),
            TestConfiguration("Full_HD_30FPS", 1920, 1080, 30, 80),
            TestConfiguration("Full_HD_High_Compression", 1920, 1080, 30, 90),
        ]
        
        # Métricas en tiempo real
        self.current_metrics = None
        self.all_results = []
        
        # Control de prueba
        self.running = False
        self.session = requests.Session()
        
        print("🧪 PerformanceTester inicializado")
        print(f"📡 Servidor: {self.server_url}")
        print(f"🎯 Configuraciones de prueba: {len(self.test_configs)}")
    
    def initialize_camera(self, width: int, height: int, fps: int) -> bool:
        """Inicializa la cámara con configuración específica."""
        try:
            if self.camera:
                self.camera.release()
            
            self.camera = CameraHandler()
            self.camera.width = width
            self.camera.height = height
            self.camera.fps = fps
            
            if not self.camera.initialize():
                print(f"❌ Error inicializando cámara {width}x{height}@{fps}fps")
                return False
            
            # Verificar resolución real
            ret, frame = self.camera.get_frame()
            if ret and frame is not None:
                actual_height, actual_width = frame.shape[:2]
                print(f"📷 Cámara inicializada: {actual_width}x{actual_height} (solicitado: {width}x{height})")
                return True
            else:
                print("❌ No se pudo capturar frame de prueba")
                return False
                
        except Exception as e:
            print(f"❌ Error en inicialización de cámara: {e}")
            return False
    
    def test_server_connection(self) -> bool:
        """Verifica conexión al servidor y mide latencia básica."""
        try:
            start_time = time.time()
            test_url = self.server_url.replace('/upload', '/stats')
            response = self.session.get(test_url, timeout=5)
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                print(f"✅ Servidor conectado - Latencia: {latency:.1f}ms")
                return True
            else:
                print(f"❌ Servidor respondió con código: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error conectando al servidor: {e}")
            return False
    
    def compress_frame(self, frame: np.ndarray, quality: int) -> Tuple[Optional[bytes], int]:
        """
        Comprime frame y retorna datos + tamaño original.
        
        Returns:
            Tuple (compressed_data, original_size_bytes)
        """
        original_size = frame.nbytes
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, encoded_frame = cv2.imencode('.jpg', frame, encode_params)
        
        if not success:
            return None, original_size
            
        return encoded_frame.tobytes(), original_size
    
    def send_frame_with_timing(self, frame_data: bytes) -> Tuple[bool, float]:
        """
        Envía frame y mide latencia.
        
        Returns:
            Tuple (success, latency_ms)
        """
        try:
            start_time = time.time()
            
            headers = {
                'Content-Type': 'application/octet-stream',
                'Content-Length': str(len(frame_data))
            }
            
            response = self.session.post(
                self.server_url,
                data=frame_data,
                headers=headers,
                timeout=2.0
            )
            
            latency_ms = (time.time() - start_time) * 1000
            success = response.status_code == 200
            
            return success, latency_ms
            
        except Exception as e:
            return False, 0.0
    
    def run_single_test(self, config: TestConfiguration) -> PerformanceMetrics:
        """
        Ejecuta una prueba individual con configuración específica.
        
        Args:
            config: Configuración de la prueba
            
        Returns:
            Métricas de rendimiento obtenidas
        """
        print(f"\n🧪 Iniciando prueba: {config.name}")
        print(f"📐 Resolución: {config.width}x{config.height}")
        print(f"🎯 FPS objetivo: {config.fps}")
        print(f"🗜️ Calidad JPEG: {config.jpeg_quality}%")
        print(f"⏱️ Duración: {config.duration_seconds}s (+ {config.warmup_seconds}s warmup)")
        
        # Inicializar cámara
        if not self.initialize_camera(config.width, config.height, config.fps):
            return PerformanceMetrics(config.name)
        
        # Inicializar métricas
        metrics = PerformanceMetrics(
            config_name=config.name,
            fps_target=config.fps
        )
        
        # Variables de control
        frame_interval = 1.0 / config.fps
        frame_sizes_kb = deque(maxlen=1000)
        latencies_ms = deque(maxlen=1000)
        cpu_readings = deque(maxlen=100)
        memory_readings = deque(maxlen=100)
        
        # Proceso para monitorear recursos del sistema
        def monitor_resources():
            while self.running:
                cpu_readings.append(psutil.cpu_percent())
                memory_readings.append(psutil.virtual_memory().used / (1024**2))
                time.sleep(0.5)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        self.running = True
        test_start_time = time.time()
        warmup_end_time = test_start_time + config.warmup_seconds
        test_end_time = warmup_end_time + config.duration_seconds
        last_frame_time = 0
        
        print("🔥 Fase de calentamiento...")
        
        try:
            while time.time() < test_end_time and self.running:
                current_time = time.time()
                
                # Control de tasa de frames
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                # Capturar frame
                ret, frame = self.camera.get_frame()
                
                if not ret or frame is None:
                    continue
                
                # Comprimir frame
                compressed_data, original_size = self.compress_frame(frame, config.jpeg_quality)
                
                if compressed_data is None:
                    continue
                
                # Enviar frame con medición de latencia
                success, latency_ms = self.send_frame_with_timing(compressed_data)
                
                # Solo contar métricas después del warmup
                is_measuring = current_time >= warmup_end_time
                
                if is_measuring:
                    if success:
                        metrics.frames_sent += 1
                        metrics.bytes_sent += len(compressed_data)
                        
                        # Recopilar métricas detalladas
                        frame_size_kb = len(compressed_data) / 1024
                        frame_sizes_kb.append(frame_size_kb)
                        latencies_ms.append(latency_ms)
                        metrics.frame_times.append(current_time)
                        
                    else:
                        metrics.frames_failed += 1
                
                last_frame_time = current_time
                
                # Feedback visual cada 2 segundos
                if int(current_time) % 2 == 0 and abs(current_time - int(current_time)) < 0.1:
                    if is_measuring:
                        elapsed = current_time - warmup_end_time
                        fps_actual = metrics.frames_sent / elapsed if elapsed > 0 else 0
                        success_rate = (metrics.frames_sent / (metrics.frames_sent + metrics.frames_failed)) * 100 if (metrics.frames_sent + metrics.frames_failed) > 0 else 0
                        print(f"📊 {elapsed:.1f}s - FPS: {fps_actual:.1f}/{config.fps} - Éxito: {success_rate:.1f}% - Latencia: {latency_ms:.1f}ms")
                    else:
                        remaining_warmup = warmup_end_time - current_time
                        print(f"🔥 Calentamiento: {remaining_warmup:.1f}s restantes")
        
        except KeyboardInterrupt:
            print("\n⚠️ Prueba interrumpida por el usuario")
        
        finally:
            self.running = False
            actual_duration = time.time() - warmup_end_time
            
            # Calcular métricas finales
            if actual_duration > 0:
                metrics.test_duration = actual_duration
                metrics.fps_actual = metrics.frames_sent / actual_duration
                metrics.throughput_mbps = (metrics.bytes_sent / actual_duration) / (1024**2)
            
            if metrics.frames_sent + metrics.frames_failed > 0:
                metrics.success_rate = (metrics.frames_sent / (metrics.frames_sent + metrics.frames_failed)) * 100
            
            if frame_sizes_kb:
                metrics.avg_frame_size_kb = np.mean(frame_sizes_kb)
                metrics.min_frame_size_kb = np.min(frame_sizes_kb)
                metrics.max_frame_size_kb = np.max(frame_sizes_kb)
            
            if latencies_ms:
                metrics.avg_latency_ms = np.mean(latencies_ms)
                metrics.min_latency_ms = np.min(latencies_ms)
                metrics.max_latency_ms = np.max(latencies_ms)
            
            if cpu_readings:
                metrics.avg_cpu_percent = np.mean(cpu_readings)
            
            if memory_readings:
                metrics.avg_memory_mb = np.mean(memory_readings)
            
            # Calcular ratio de compresión
            if metrics.frames_sent > 0:
                original_total_mb = (config.width * config.height * 3 * metrics.frames_sent) / (1024**2)
                compressed_total_mb = metrics.bytes_sent / (1024**2)
                metrics.compression_ratio = original_total_mb / compressed_total_mb if compressed_total_mb > 0 else 0
        
        return metrics
    
    def print_test_results(self, metrics: PerformanceMetrics):
        """Imprime resultados detallados de una prueba."""
        print(f"\n📊 RESULTADOS - {metrics.config_name}")
        print("=" * 60)
        
        # Transmisión
        print(f"📤 Frames enviados: {metrics.frames_sent}")
        print(f"❌ Frames fallidos: {metrics.frames_failed}")
        print(f"✅ Tasa de éxito: {metrics.success_rate:.1f}%")
        
        # FPS y timing
        print(f"🎯 FPS objetivo: {metrics.fps_target:.1f}")
        print(f"📊 FPS real: {metrics.fps_actual:.1f}")
        fps_efficiency = (metrics.fps_actual / metrics.fps_target) * 100 if metrics.fps_target > 0 else 0
        print(f"⚡ Eficiencia FPS: {fps_efficiency:.1f}%")
        
        # Tamaños y throughput
        print(f"📦 Tamaño promedio: {metrics.avg_frame_size_kb:.1f} KB/frame")
        print(f"📦 Rango tamaños: {metrics.min_frame_size_kb:.1f} - {metrics.max_frame_size_kb:.1f} KB")
        print(f"🚀 Throughput: {metrics.throughput_mbps:.2f} MB/s")
        print(f"🗜️ Ratio compresión: {metrics.compression_ratio:.1f}x")
        
        # Latencia
        print(f"⏱️ Latencia promedio: {metrics.avg_latency_ms:.1f}ms")
        print(f"⏱️ Rango latencia: {metrics.min_latency_ms:.1f} - {metrics.max_latency_ms:.1f}ms")
        
        # Recursos
        print(f"💻 CPU promedio: {metrics.avg_cpu_percent:.1f}%")
        print(f"🧠 Memoria promedio: {metrics.avg_memory_mb:.1f} MB")
        
        # Evaluación general
        print(f"\n🏆 EVALUACIÓN GENERAL:")
        
        if fps_efficiency >= 95:
            print("✅ Excelente - FPS estable y eficiente")
        elif fps_efficiency >= 85:
            print("🟢 Bueno - FPS aceptable con ligeras variaciones")
        elif fps_efficiency >= 70:
            print("🟡 Regular - FPS bajo el objetivo, considerar optimizaciones")
        else:
            print("🔴 Pobre - FPS muy bajo, requiere ajustes significativos")
        
        if metrics.success_rate >= 99:
            print("✅ Excelente - Transmisión muy confiable")
        elif metrics.success_rate >= 95:
            print("🟢 Bueno - Transmisión confiable")
        elif metrics.success_rate >= 90:
            print("🟡 Regular - Algunas pérdidas de frames")
        else:
            print("🔴 Pobre - Muchas pérdidas, verificar red")
        
        if metrics.avg_latency_ms <= 50:
            print("✅ Excelente - Latencia muy baja")
        elif metrics.avg_latency_ms <= 100:
            print("🟢 Bueno - Latencia aceptable")
        elif metrics.avg_latency_ms <= 200:
            print("🟡 Regular - Latencia notable")
        else:
            print("🔴 Pobre - Latencia muy alta")
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas de rendimiento configuradas."""
        print("🚀 INICIANDO BATERÍA DE PRUEBAS DE RENDIMIENTO")
        print("=" * 60)
        
        if not self.test_server_connection():
            print("❌ No se puede conectar al servidor. Abortando pruebas.")
            return
        
        self.all_results = []
        
        for i, config in enumerate(self.test_configs, 1):
            print(f"\n🔄 Prueba {i}/{len(self.test_configs)}")
            
            try:
                metrics = self.run_single_test(config)
                self.all_results.append(metrics)
                self.print_test_results(metrics)
                
                # Pausa entre pruebas
                if i < len(self.test_configs):
                    print(f"\n⏸️ Pausa de 5 segundos antes de la siguiente prueba...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"❌ Error en prueba {config.name}: {e}")
                continue
        
        # Generar resumen final
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Genera un reporte resumen de todas las pruebas."""
        if not self.all_results:
            print("❌ No hay resultados para generar reporte")
            return
        
        print("\n" + "=" * 80)
        print("📈 REPORTE RESUMEN - COMPARACIÓN DE CONFIGURACIONES")
        print("=" * 80)
        
        # Tabla de comparación
        print(f"{'Configuración':<20} {'FPS Real':<10} {'Éxito%':<8} {'Throughput':<12} {'Latencia':<10} {'CPU%':<6}")
        print("-" * 80)
        
        best_config = None
        best_score = 0
        
        for metrics in self.all_results:
            fps_eff = (metrics.fps_actual / metrics.fps_target) * 100 if metrics.fps_target > 0 else 0
            
            # Calcular puntuación compuesta
            score = (fps_eff * 0.4) + (metrics.success_rate * 0.3) + (min(100, 1000/max(1, metrics.avg_latency_ms)) * 0.3)
            
            if score > best_score:
                best_score = score
                best_config = metrics
            
            print(f"{metrics.config_name:<20} {metrics.fps_actual:<10.1f} {metrics.success_rate:<8.1f} {metrics.throughput_mbps:<12.2f} {metrics.avg_latency_ms:<10.1f} {metrics.avg_cpu_percent:<6.1f}")
        
        print("\n🏆 RECOMENDACIONES:")
        if best_config:
            print(f"✅ Mejor configuración: {best_config.config_name}")
            print(f"   - FPS: {best_config.fps_actual:.1f} ({(best_config.fps_actual/best_config.fps_target)*100:.1f}% eficiencia)")
            print(f"   - Throughput: {best_config.throughput_mbps:.2f} MB/s")
            print(f"   - Latencia: {best_config.avg_latency_ms:.1f}ms")
        
        # Guardar resultados en JSON
        self.save_results_to_file()
    
    def save_results_to_file(self):
        """Guarda los resultados en un archivo JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_test_{timestamp}.json"
        
        # Preparar datos para JSON
        results_data = {
            'timestamp': timestamp,
            'server_url': self.server_url,
            'results': []
        }
        
        for metrics in self.all_results:
            result_data = {
                'config_name': metrics.config_name,
                'frames_sent': metrics.frames_sent,
                'frames_failed': metrics.frames_failed,
                'test_duration': metrics.test_duration,
                'fps_actual': metrics.fps_actual,
                'fps_target': metrics.fps_target,
                'avg_frame_size_kb': metrics.avg_frame_size_kb,
                'throughput_mbps': metrics.throughput_mbps,
                'avg_latency_ms': metrics.avg_latency_ms,
                'success_rate': metrics.success_rate,
                'compression_ratio': metrics.compression_ratio,
                'avg_cpu_percent': metrics.avg_cpu_percent,
                'avg_memory_mb': metrics.avg_memory_mb
            }
            results_data['results'].append(result_data)
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"💾 Resultados guardados en: {filename}")
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
    
    def cleanup(self):
        """Limpia recursos."""
        self.running = False
        if self.camera:
            self.camera.release()
        self.session.close()


def main():
    """Función principal del probador de rendimiento."""
    print("🧪 PROBADOR DE RENDIMIENTO - VIDEO STREAMING")
    print("Evaluando capacidad para 1920×1080 en tiempo real")
    print("=" * 60)
    
    # Crear probador
    tester = PerformanceTester()
    
    try:
        # Ejecutar todas las pruebas
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️ Pruebas interrumpidas por el usuario")
    
    finally:
        tester.cleanup()
        print("\n👋 Pruebas completadas")


if __name__ == "__main__":
    main()
