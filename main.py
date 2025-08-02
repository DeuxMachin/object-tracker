
"""
Object Tracking System - Main Entry Point

Sistema modular de seguimiento de objetos con arquitectura cliente-servidor.
- Cliente: Captura de video desde cámaras web
- Servidor: Procesamiento, detección y tracking de objetos
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


try:
    from client.camera_handler import test_camera
    print("✅ Módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error al importar módulos: {e}")
    print("💡 Verifica que todas las dependencias estén instaladas")
    print("   pip install opencv-python numpy")
    sys.exit(1)


def main():
    """Función principal del sistema."""
    
    # Banner del sistema
    print("=" * 60)
    print("� OBJECT TRACKING SYSTEM")
    print("   Sistema Modular de Seguimiento de Objetos")
    print("=" * 60)
    print()
    print("📋 ARQUITECTURA:")
    print("   • CLIENT:  Captura de video (cámara web)")
    print("   • SERVER:  Procesamiento y tracking")
    print("   • SHARED:  Configuración común")
    print()
    print("📹 MODO ACTUAL: Test de Cámara (Cliente)")
    print("🔜 PRÓXIMO: Integración con servidor de detección")
    print("=" * 60)
    print()
    print("💡 INSTRUCCIONES:")
    print("   • Se abrirá una ventana con el video de tu cámara")
    print("   • Presiona 'q' en la ventana para salir")
    print()
    
    
    try:
        success = test_camera()
        
        if success:
            print()
            print(" ¡ÉXITO! Tu cámara funciona perfectamente")
        else:
            print()
            print(" La cámara no funcionó como esperábamos")
           
    except KeyboardInterrupt:
        print()
        print("⚠️  Programa interrumpido por el usuario")
        print("👋 ¡Hasta la próxima!")
        






if __name__ == "__main__":
    main()
