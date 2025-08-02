

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


try:
    from src.video.camera_handler import test_camera
    print("✅ Todos los módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error al importar módulos: {e}")
    print()
    sys.exit(1)


def main():

    
    # 🎨 Banner de bienvenida
    print("=" * 60)
    print("🎥 SISTEMA DE SEGUIMIENTO DE OBJETOS")
    print("   ¡Bienvenido al futuro de la visión computacional!")
    print("=" * 60)
    print()
    print("📹 MODO ACTUAL: Prueba de Cámara Web")
    print("🎯 PRÓXIMAMENTE: Detección y seguimiento inteligente")
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
