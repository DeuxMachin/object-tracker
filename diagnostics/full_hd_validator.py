"""
Full HD Performance Validator - Validación Específica para 1920×1080

Script optimizado para validar la capacidad del sistema para transmitir
video Full HD (1920×1080) en tiempo real con diferentes configuraciones.

Este script está diseñado específicamente para:
- Evaluar el throughput necesario para Full HD
- Validar que el ancho de banda de 475 Mbps (subida) y 640 Mbps (bajada) es suficiente
- Encontrar la configuración óptima de FPS y compresión
- Proporcionar recomendaciones específicas para tu hardware

Ejecutar: python full_hd_validator.py
"""

import sys
import os
import time
import json
from datetime import datetime

# Agregar directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.performance_test import PerformanceTester, TestConfiguration, PerformanceMetrics

class FullHDValidator:
    """
    Validador específico para capacidades Full HD (1920×1080).
    
    Ejecuta pruebas progresivas para encontrar la configuración óptima
    que aproveche al máximo el ancho de banda disponible.
    """
    
    def __init__(self, server_url: str = None):
        """Inicializa el validador Full HD."""
        self.tester = PerformanceTester(server_url)
        
        # Ancho de banda disponible (según especificaciones del usuario)
        self.upload_mbps = 475  # Mbps del cliente (Equipo B)
        self.download_mbps = 640  # Mbps del servidor (Equipo A)
        self.effective_bandwidth_mbs = min(self.upload_mbps, self.download_mbps) / 8  # Convertir a MB/s
        
        print("🎯 VALIDADOR FULL HD - 1920×1080")
        print("=" * 50)
        print(f"📤 Ancho banda subida: {self.upload_mbps} Mbps")
        print(f"📥 Ancho banda bajada: {self.download_mbps} Mbps")
        print(f"🚀 Bandwidth efectivo: {self.effective_bandwidth_mbs:.1f} MB/s")
        print(f"💡 Teóricamente suficiente para >59 MB/s Full HD")
    
    def calculate_theoretical_limits(self):
        """Calcula los límites teóricos para Full HD."""
        width, height = 1920, 1080
        bytes_per_pixel = 3  # RGB
        
        # Sin compresión
        raw_bytes_per_frame = width * height * bytes_per_pixel
        raw_mb_per_frame = raw_bytes_per_frame / (1024 * 1024)
        
        # FPS máximo teórico sin compresión
        max_fps_raw = self.effective_bandwidth_mbs / raw_mb_per_frame
        
        print(f"\n📊 LÍMITES TEÓRICOS PARA 1920×1080:")
        print(f"   📦 Tamaño raw por frame: {raw_mb_per_frame:.2f} MB")
        print(f"   🎬 FPS máximo sin compresión: {max_fps_raw:.1f}")
        
        # Con diferentes niveles de compresión JPEG
        compression_ratios = {
            95: 0.20,  # Calidad muy alta
            85: 0.15,  # Calidad alta
            75: 0.10,  # Calidad media-alta
            65: 0.08,  # Calidad media
            55: 0.06,  # Calidad media-baja
        }
        
        print(f"\n🗜️ CON COMPRESIÓN JPEG:")
        for quality, ratio in compression_ratios.items():
            compressed_mb_per_frame = raw_mb_per_frame * ratio
            max_fps_compressed = self.effective_bandwidth_mbs / compressed_mb_per_frame
            bandwidth_at_30fps = compressed_mb_per_frame * 30
            
            print(f"   Q{quality}%: {compressed_mb_per_frame:.3f} MB/frame → "
                  f"máx {max_fps_compressed:.1f} FPS → "
                  f"{bandwidth_at_30fps:.1f} MB/s @ 30fps")
        
        return raw_mb_per_frame, compression_ratios
    
    def create_progressive_test_configs(self) -> list:
        """Crea configuraciones de prueba progresivas para Full HD."""
        configs = []
        
        # Prueba 1: Configuración conservadora
        configs.append(TestConfiguration(
            name="FullHD_Conservative",
            width=1920, height=1080,
            fps=15, jpeg_quality=70,
            duration_seconds=60
        ))
        
        # Prueba 2: Configuración balanceada
        configs.append(TestConfiguration(
            name="FullHD_Balanced", 
            width=1920, height=1080,
            fps=20, jpeg_quality=75,
            duration_seconds=90
        ))
        
        # Prueba 3: Configuración óptima
        configs.append(TestConfiguration(
            name="FullHD_Optimal",
            width=1920, height=1080,
            fps=25, jpeg_quality=80,
            duration_seconds=120
        ))
        
        # Prueba 4: Máximo rendimiento
        configs.append(TestConfiguration(
            name="FullHD_Maximum",
            width=1920, height=1080,
            fps=30, jpeg_quality=85,
            duration_seconds=120
        ))
        
        # Prueba 5: Calidad máxima
        configs.append(TestConfiguration(
            name="FullHD_MaxQuality",
            width=1920, height=1080,
            fps=30, jpeg_quality=95,
            duration_seconds=90
        ))
        
        return configs
    
    def evaluate_config_viability(self, config: TestConfiguration) -> dict:
        """Evalúa la viabilidad teórica de una configuración."""
        raw_mb_per_frame = (config.width * config.height * 3) / (1024 * 1024)
        
        # Estimar compresión basada en calidad JPEG
        compression_factor = max(0.04, min(0.25, (100 - config.jpeg_quality) / 400 + 0.05))
        estimated_mb_per_frame = raw_mb_per_frame * compression_factor
        estimated_throughput = estimated_mb_per_frame * config.fps
        
        bandwidth_usage = (estimated_throughput / self.effective_bandwidth_mbs) * 100
        
        return {
            'estimated_throughput_mbs': estimated_throughput,
            'bandwidth_usage_percent': bandwidth_usage,
            'is_viable': bandwidth_usage < 80,  # Dejar 20% de margen
            'compression_factor': compression_factor
        }
    
    def run_validation(self):
        """Ejecuta la validación completa para Full HD."""
        print("\n🚀 INICIANDO VALIDACIÓN FULL HD")
        print("=" * 50)
        
        # Mostrar límites teóricos
        self.calculate_theoretical_limits()
        
        # Verificar conexión al servidor
        if not self.tester.test_server_connection():
            print("❌ No se puede conectar al servidor. Abortando validación.")
            return
        
        # Crear configuraciones de prueba
        configs = self.create_progressive_test_configs()
        results = []
        
        print(f"\n📋 EJECUTANDO {len(configs)} PRUEBAS PROGRESIVAS")
        
        for i, config in enumerate(configs, 1):
            print(f"\n🔄 PRUEBA {i}/{len(configs)}: {config.name}")
            
            # Evaluación teórica previa
            viability = self.evaluate_config_viability(config)
            print(f"🧮 Análisis teórico:")
            print(f"   Throughput estimado: {viability['estimated_throughput_mbs']:.2f} MB/s")
            print(f"   Uso de bandwidth: {viability['bandwidth_usage_percent']:.1f}%")
            print(f"   Viable: {'✅ Sí' if viability['is_viable'] else '❌ No'}")
            
            if not viability['is_viable']:
                print("⚠️ Configuración posiblemente inviable, pero probando de todos modos...")
            
            try:
                # Ejecutar prueba real
                metrics = self.tester.run_single_test(config)
                metrics.theoretical_analysis = viability
                results.append(metrics)
                
                # Análisis inmediato
                self.analyze_single_result(metrics)
                
                # Pausa entre pruebas
                if i < len(configs):
                    print(f"\n⏸️ Pausa de 10 segundos antes de la siguiente prueba...")
                    time.sleep(10)
                    
            except Exception as e:
                print(f"❌ Error en prueba {config.name}: {e}")
                continue
        
        # Análisis final y recomendaciones
        self.generate_final_recommendations(results)
        
        # Guardar resultados
        self.save_validation_results(results)
    
    def analyze_single_result(self, metrics: PerformanceMetrics):
        """Analiza los resultados de una prueba individual."""
        print(f"\n📊 ANÁLISIS: {metrics.config_name}")
        print("-" * 40)
        
        # Eficiencia de FPS
        fps_efficiency = (metrics.fps_actual / metrics.fps_target) * 100 if metrics.fps_target > 0 else 0
        
        # Comparación teórica vs real
        theoretical = metrics.theoretical_analysis
        actual_vs_theoretical = (metrics.throughput_mbps / theoretical['estimated_throughput_mbs']) * 100 if theoretical['estimated_throughput_mbs'] > 0 else 0
        
        print(f"🎯 FPS: {metrics.fps_actual:.1f}/{metrics.fps_target} ({fps_efficiency:.1f}% eficiencia)")
        print(f"📦 Tamaño promedio frame: {metrics.avg_frame_size_kb:.1f} KB")
        print(f"🚀 Throughput: {metrics.throughput_mbps:.2f} MB/s")
        print(f"✅ Tasa de éxito: {metrics.success_rate:.1f}%")
        print(f"⏱️ Latencia promedio: {metrics.avg_latency_ms:.1f}ms")
        print(f"📊 Real vs Teórico: {actual_vs_theoretical:.1f}%")
        
        # Evaluación de calidad
        if fps_efficiency >= 95 and metrics.success_rate >= 99:
            verdict = "🟢 EXCELENTE - Configuración muy estable"
        elif fps_efficiency >= 85 and metrics.success_rate >= 95:
            verdict = "🟡 BUENA - Configuración aceptable"
        elif fps_efficiency >= 70 and metrics.success_rate >= 90:
            verdict = "🟠 REGULAR - Necesita optimización"
        else:
            verdict = "🔴 POBRE - No recomendada"
        
        print(f"🏆 {verdict}")
        
        # Utilización del ancho de banda
        bandwidth_usage = (metrics.throughput_mbps / self.effective_bandwidth_mbs) * 100
        print(f"📈 Uso del ancho de banda: {bandwidth_usage:.1f}%")
        
        if bandwidth_usage < 50:
            print("💡 Bandwidth subutilizado - Se puede aumentar calidad/FPS")
        elif bandwidth_usage < 80:
            print("✅ Uso óptimo del bandwidth")
        else:
            print("⚠️ Bandwidth al límite - Riesgo de congestión")
    
    def generate_final_recommendations(self, results: list):
        """Genera recomendaciones finales basadas en todos los resultados."""
        if not results:
            print("❌ No hay resultados para analizar")
            return
        
        print("\n" + "=" * 80)
        print("🏆 RECOMENDACIONES FINALES PARA FULL HD (1920×1080)")
        print("=" * 80)
        
        # Encontrar la mejor configuración
        best_config = None
        best_score = 0
        
        viable_configs = []
        
        print(f"{'Configuración':<20} {'FPS':<8} {'Throughput':<12} {'Éxito%':<8} {'Latencia':<10} {'Veredicto':<15}")
        print("-" * 85)
        
        for metrics in results:
            fps_eff = (metrics.fps_actual / metrics.fps_target) * 100 if metrics.fps_target > 0 else 0
            
            # Puntuación compuesta (FPS 40%, éxito 30%, latencia 20%, throughput 10%)
            latency_score = min(100, 1000 / max(1, metrics.avg_latency_ms))
            throughput_score = min(100, (metrics.throughput_mbps / 10) * 100)  # Normalizar a 10 MB/s
            
            score = (fps_eff * 0.4) + (metrics.success_rate * 0.3) + (latency_score * 0.2) + (throughput_score * 0.1)
            
            # Determinar veredicto
            if fps_eff >= 95 and metrics.success_rate >= 99:
                verdict = "EXCELENTE"
                viable_configs.append(metrics)
            elif fps_eff >= 85 and metrics.success_rate >= 95:
                verdict = "BUENA"
                viable_configs.append(metrics)
            elif fps_eff >= 70 and metrics.success_rate >= 90:
                verdict = "REGULAR"
            else:
                verdict = "POBRE"
            
            if score > best_score:
                best_score = score
                best_config = metrics
            
            print(f"{metrics.config_name:<20} {metrics.fps_actual:<8.1f} {metrics.throughput_mbps:<12.2f} {metrics.success_rate:<8.1f} {metrics.avg_latency_ms:<10.1f} {verdict:<15}")
        
        print("\n🎯 RECOMENDACIÓN PRINCIPAL:")
        if best_config:
            print(f"✅ Mejor configuración: {best_config.config_name}")
            print(f"   📊 FPS real: {best_config.fps_actual:.1f}")
            print(f"   🚀 Throughput: {best_config.throughput_mbps:.2f} MB/s")
            print(f"   ✅ Tasa de éxito: {best_config.success_rate:.1f}%")
            print(f"   ⏱️ Latencia: {best_config.avg_latency_ms:.1f}ms")
            
            bandwidth_usage = (best_config.throughput_mbps / self.effective_bandwidth_mbs) * 100
            print(f"   📈 Uso de bandwidth: {bandwidth_usage:.1f}%")
        
        print(f"\n📋 CONFIGURACIONES VIABLES: {len(viable_configs)}")
        for config in viable_configs:
            fps_eff = (config.fps_actual / config.fps_target) * 100
            print(f"   • {config.config_name}: {config.fps_actual:.1f} FPS ({fps_eff:.1f}% eficiencia)")
        
        print(f"\n💡 CONCLUSIONES:")
        
        # Análisis del ancho de banda
        max_throughput = max([r.throughput_mbps for r in results])
        bandwidth_utilization = (max_throughput / self.effective_bandwidth_mbs) * 100
        
        if bandwidth_utilization < 30:
            print(f"🟢 Su conexión ({self.effective_bandwidth_mbs:.1f} MB/s) es más que suficiente para Full HD")
            print(f"   Puede usar configuraciones de máxima calidad sin problemas")
        elif bandwidth_utilization < 70:
            print(f"🟡 Su conexión es adecuada para Full HD con margen de seguridad")
            print(f"   Recomendamos configuraciones balanceadas u óptimas")
        else:
            print(f"🟠 Su conexión está cerca del límite para Full HD")
            print(f"   Use configuraciones conservadoras para estabilidad")
        
        # Análisis de hardware
        if best_config and best_config.avg_cpu_percent > 80:
            print(f"⚠️ Alto uso de CPU ({best_config.avg_cpu_percent:.1f}%) - Considere reducir FPS o calidad")
        
        if len(viable_configs) >= 3:
            print(f"🟢 Su sistema maneja bien múltiples configuraciones Full HD")
        elif len(viable_configs) >= 1:
            print(f"🟡 Su sistema puede manejar Full HD con limitaciones")
        else:
            print(f"🔴 Su sistema tiene dificultades con Full HD - Considere HD 720p")
    
    def save_validation_results(self, results: list):
        """Guarda los resultados de validación en un archivo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"full_hd_validation_{timestamp}.json"
        
        validation_data = {
            'timestamp': timestamp,
            'server_url': self.tester.server_url,
            'network_specs': {
                'upload_mbps': self.upload_mbps,
                'download_mbps': self.download_mbps,
                'effective_bandwidth_mbs': self.effective_bandwidth_mbs
            },
            'results': []
        }
        
        for metrics in results:
            result_data = {
                'config_name': metrics.config_name,
                'fps_target': metrics.fps_target,
                'fps_actual': metrics.fps_actual,
                'fps_efficiency': (metrics.fps_actual / metrics.fps_target) * 100 if metrics.fps_target > 0 else 0,
                'frames_sent': metrics.frames_sent,
                'frames_failed': metrics.frames_failed,
                'success_rate': metrics.success_rate,
                'throughput_mbps': metrics.throughput_mbps,
                'avg_frame_size_kb': metrics.avg_frame_size_kb,
                'avg_latency_ms': metrics.avg_latency_ms,
                'compression_ratio': metrics.compression_ratio,
                'avg_cpu_percent': metrics.avg_cpu_percent,
                'bandwidth_usage_percent': (metrics.throughput_mbps / self.effective_bandwidth_mbs) * 100,
                'theoretical_analysis': getattr(metrics, 'theoretical_analysis', {})
            }
            validation_data['results'].append(result_data)
        
        try:
            with open(filename, 'w') as f:
                json.dump(validation_data, f, indent=2)
            print(f"\n💾 Resultados de validación guardados en: {filename}")
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
    
    def cleanup(self):
        """Limpia recursos."""
        self.tester.cleanup()


def main():
    """Función principal del validador Full HD."""
    print("🎯 VALIDADOR FULL HD - SISTEMA DE STREAMING 1920×1080")
    print("Optimizado para redes LAN de alta velocidad")
    print("=" * 60)
    
    validator = FullHDValidator()
    
    try:
        validator.run_validation()
    except KeyboardInterrupt:
        print("\n⚠️ Validación interrumpida por el usuario")
    finally:
        validator.cleanup()
        print("\n👋 Validación completada")


if __name__ == "__main__":
    main()
