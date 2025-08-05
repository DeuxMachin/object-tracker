"""
Quick Test - Pruebas Rápidas de Rendimiento

Script para ejecutar pruebas específicas y rápidas del sistema de streaming.
Ideal para verificar configuraciones sin ejecutar la batería completa de pruebas.

Uso:
    python quick_test.py                    # Prueba rápida por defecto
    python quick_test.py --profile full_hd  # Prueba específica Full HD
    python quick_test.py --custom 1920 1080 30 80  # Prueba personalizada
"""

import sys
import os
import argparse
import time

# Agregar directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.performance_test import PerformanceTester, TestConfiguration
from diagnostics.test_profiles import get_profile_by_name, PERFORMANCE_PROFILES, print_all_profiles

def create_quick_configs():
    """Crea configuraciones rápidas predefinidas."""
    return {
        'connectivity': TestConfiguration(
            name="Quick_Connectivity",
            width=640, height=480, fps=10, jpeg_quality=60,
            duration_seconds=15, warmup_seconds=3
        ),
        'hd': TestConfiguration(
            name="Quick_HD",
            width=1280, height=720, fps=20, jpeg_quality=75,
            duration_seconds=30, warmup_seconds=5
        ),
        'full_hd': TestConfiguration(
            name="Quick_FullHD",
            width=1920, height=1080, fps=25, jpeg_quality=80,
            duration_seconds=45, warmup_seconds=10
        ),
        'max_quality': TestConfiguration(
            name="Quick_MaxQuality",
            width=1920, height=1080, fps=30, jpeg_quality=95,
            duration_seconds=30, warmup_seconds=5
        )
    }

def run_quick_test(config: TestConfiguration, server_url: str = None):
    """Ejecuta una prueba rápida con configuración específica."""
    print(f"🚀 PRUEBA RÁPIDA: {config.name}")
    print("=" * 50)
    
    tester = PerformanceTester(server_url)
    
    try:
        # Verificar conexión
        if not tester.test_server_connection():
            print("❌ No se puede conectar al servidor")
            return False
        
        # Ejecutar prueba
        metrics = tester.run_single_test(config)
        
        # Mostrar resultados resumidos
        print_quick_results(metrics)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        return False
    
    finally:
        tester.cleanup()

def print_quick_results(metrics):
    """Imprime resultados resumidos de una prueba rápida."""
    print(f"\n📊 RESULTADOS RÁPIDOS - {metrics.config_name}")
    print("=" * 50)
    
    # Métricas principales
    fps_efficiency = (metrics.fps_actual / metrics.fps_target) * 100 if metrics.fps_target > 0 else 0
    
    print(f"🎯 FPS: {metrics.fps_actual:.1f}/{metrics.fps_target} ({fps_efficiency:.1f}% eficiencia)")
    print(f"✅ Tasa de éxito: {metrics.success_rate:.1f}%")
    print(f"📦 Tamaño promedio: {metrics.avg_frame_size_kb:.1f} KB/frame")
    print(f"🚀 Throughput: {metrics.throughput_mbps:.2f} MB/s")
    print(f"⏱️ Latencia: {metrics.avg_latency_ms:.1f}ms")
    print(f"💻 CPU promedio: {metrics.avg_cpu_percent:.1f}%")
    
    # Evaluación rápida
    if fps_efficiency >= 95 and metrics.success_rate >= 99:
        status = "🟢 EXCELENTE"
        recommendation = "Configuración muy estable, apta para producción"
    elif fps_efficiency >= 85 and metrics.success_rate >= 95:
        status = "🟡 BUENA"
        recommendation = "Configuración aceptable para uso general"
    elif fps_efficiency >= 70 and metrics.success_rate >= 90:
        status = "🟠 REGULAR"
        recommendation = "Funciona pero podría necesitar optimización"
    else:
        status = "🔴 POBRE"
        recommendation = "No recomendada, considerar configuración más baja"
    
    print(f"\n🏆 EVALUACIÓN: {status}")
    print(f"💡 {recommendation}")
    
    # Estimación de ancho de banda
    theoretical_max_mbps = metrics.throughput_mbps * 1.2  # Con 20% de margen
    print(f"\n📈 Ancho de banda recomendado: {theoretical_max_mbps:.1f} MB/s ({theoretical_max_mbps*8:.0f} Mbps)")

def main():
    """Función principal del quick test."""
    parser = argparse.ArgumentParser(description='Pruebas rápidas de rendimiento para streaming')
    
    parser.add_argument('--profile', choices=['connectivity', 'hd', 'full_hd', 'max_quality'],
                       default='connectivity',
                       help='Perfil de prueba predefinido (default: connectivity)')
    
    parser.add_argument('--custom', nargs=4, metavar=('WIDTH', 'HEIGHT', 'FPS', 'QUALITY'),
                       type=int, help='Configuración personalizada: ancho alto fps calidad')
    
    parser.add_argument('--server', type=str, default=None,
                       help='URL del servidor (default: desde config)')
    
    parser.add_argument('--duration', type=int, default=None,
                       help='Duración de la prueba en segundos')
    
    parser.add_argument('--list-profiles', action='store_true',
                       help='Mostrar todos los perfiles disponibles')
    
    args = parser.parse_args()
    
    # Mostrar perfiles si se solicita
    if args.list_profiles:
        print_all_profiles()
        return
    
    print("⚡ QUICK TEST - PRUEBAS RÁPIDAS DE RENDIMIENTO")
    print("Para análisis completo, use performance_test.py o full_hd_validator.py")
    print("=" * 70)
    
    # Configurar prueba
    if args.custom:
        width, height, fps, quality = args.custom
        config = TestConfiguration(
            name=f"Custom_{width}x{height}_{fps}fps_Q{quality}",
            width=width, height=height, fps=fps, jpeg_quality=quality,
            duration_seconds=args.duration or 30, warmup_seconds=5
        )
        print(f"🔧 Configuración personalizada: {width}x{height} @ {fps}fps, calidad {quality}%")
    
    else:
        quick_configs = create_quick_configs()
        config = quick_configs[args.profile]
        
        if args.duration:
            config.duration_seconds = args.duration
        
        print(f"📋 Perfil seleccionado: {args.profile}")
    
    print(f"⏱️ Duración: {config.duration_seconds}s (warmup: {config.warmup_seconds}s)")
    print(f"🎯 Resolución: {config.width}x{config.height}")
    print(f"📺 FPS objetivo: {config.fps}")
    print(f"🗜️ Calidad JPEG: {config.jpeg_quality}%")
    
    # Ejecutar prueba
    success = run_quick_test(config, args.server)
    
    if success:
        print("\n✅ Prueba completada exitosamente")
        print("💡 Para pruebas más exhaustivas:")
        print("   - full_hd_validator.py (específico para 1920×1080)")
        print("   - performance_test.py (batería completa)")
    else:
        print("\n❌ La prueba falló")
        print("🔧 Verifique:")
        print("   - Que el servidor esté ejecutándose")
        print("   - La configuración de red")
        print("   - Que la cámara esté disponible")

if __name__ == "__main__":
    main()
