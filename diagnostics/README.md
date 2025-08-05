# 🧪 Módulo de Diagnósticos de Rendimiento

Este módulo proporciona herramientas especializadas para evaluar y optimizar el rendimiento del sistema de streaming de video cliente-servidor, con enfoque especial en transmisión Full HD (1920×1080).

## 📁 Archivos del Módulo

### Main Scripts

- **`full_hd_validator.py`** - Validador especializado para Full HD
- **`performance_test.py`** - Suite completa de pruebas de rendimiento  
- **`quick_test.py`** - Pruebas rápidas y específicas
- **`test_profiles.py`** - Perfiles y configuraciones predefinidas

## Usage Guide

### Validación Full HD (Recomendado para tu caso)

Para evaluar específicamente la capacidad de transmitir 1920×1080 a 30 FPS:

```bash
python full_hd_validator.py
```

**Características:**
- Progressive evaluation of 5 Full HD configurations
- Theoretical vs real performance analysis
- Specific metrics: FPS, throughput, latency, CPU usage
- Personalized recommendations based on your bandwidth (475/640 Mbps)
- Empirical validation for >59 MB/s throughput

### Pruebas Rápidas

Para verificaciones rápidas de configuraciones específicas:

```bash
# Prueba rápida Full HD (45 segundos)
python quick_test.py --profile full_hd

# Configuración personalizada
python quick_test.py --custom 1920 1080 30 85

# Ver todas las opciones
python quick_test.py --list-profiles
```

### Suite Completa de Rendimiento

Para análisis exhaustivo con múltiples resoluciones:

```bash
python performance_test.py
```

## Evaluated Metrics

### 🎬 Rendimiento de Video
- **FPS Real vs Objetivo** - Estabilidad del framerate
- **Tamaño Promedio de Frame** - Eficiencia de compresión
- **Ratio de Compresión** - Factor de reducción JPEG

### 🌐 Rendimiento de Red
- **Throughput (MB/s)** - Ancho de banda utilizado
- **Tasa de Éxito (%)** - Frames transmitidos exitosamente
- **Latencia (ms)** - Tiempo de ida y vuelta

### 💻 Recursos del Sistema
- **Uso de CPU (%)** - Carga de procesamiento
- **Uso de Memoria (MB)** - Consumo de RAM
- **Eficiencia de Bandwidth** - Utilización del ancho de banda disponible

## Predefined Configurations

### Full HD Progressives
1. **Conservative** - 1920×1080, 15 FPS, 70% calidad
2. **Balanced** - 1920×1080, 20 FPS, 75% calidad
3. **Optimal** - 1920×1080, 25 FPS, 80% calidad
4. **Maximum** - 1920×1080, 30 FPS, 85% calidad
5. **Max Quality** - 1920×1080, 30 FPS, 95% calidad

### Otras Resoluciones
- **Standard** - 640×480 para equipos limitados
- **HD** - 1280×720 para streaming regular
- **Connectivity** - 320×240 para pruebas básicas

## Results Interpretation

### 🟢 Excelente (Recomendado para producción)
- FPS Eficiencia: ≥95%
- Tasa de Éxito: ≥99%
- Latencia: ≤50ms

### 🟡 Bueno (Aceptable para uso general)
- FPS Eficiencia: 85-94%
- Tasa de Éxito: 95-98%
- Latencia: 51-100ms

### 🟠 Regular (Necesita optimización)
- FPS Eficiencia: 70-84%
- Tasa de Éxito: 90-94%
- Latencia: 101-200ms

### 🔴 Pobre (No recomendado)
- FPS Eficiencia: <70%
- Tasa de Éxito: <90%
- Latencia: >200ms

## Specific Recommendations for Your Setup

### Ancho de Banda Disponible
- **Subida (Equipo B):** 475 Mbps (59.3 MB/s)
- **Bajada (Equipo A):** 640 Mbps (80 MB/s)
- **Efectivo:** 59.3 MB/s (limitado por subida)

### Configuraciones Recomendadas
Con tu ancho de banda, podrías manejar teóricamente:

1. **Full HD 30 FPS + Calidad Alta (85%)** - ~8-12 MB/s
2. **Full HD 30 FPS + Calidad Máxima (95%)** - ~15-20 MB/s
3. **Multiple streams simultáneos** - Hasta 3-4 streams Full HD

### Optimizaciones Sugeridas
- Usar configuración "Maximum" o "Max Quality" para aprovechar el ancho de banda
- Monitorear uso de CPU para evitar cuellos de botella de procesamiento
- Considerar múltiples streams si el proyecto lo requiere

## 🔧 Requisitos del Sistema

### Dependencias Python
```bash
pip install opencv-python requests psutil numpy flask
```

### Hardware Mínimo Recomendado
- **CPU:** Quad-core moderna (para Full HD 30fps)
- **RAM:** 8GB+ 
- **Cámara:** Compatible con 1920×1080 @ 30fps
- **Red:** LAN Gigabit o WiFi AC

## 📝 Archivos de Salida

### Resultados JSON
Los tests generan archivos JSON con resultados detallados:
- `performance_test_YYYYMMDD_HHMMSS.json`
- `full_hd_validation_YYYYMMDD_HHMMSS.json`

### Estructura de Datos
```json
{
  "timestamp": "20250805_143022",
  "server_url": "http://192.168.1.84:5000/upload",
  "network_specs": {
    "upload_mbps": 475,
    "download_mbps": 640,
    "effective_bandwidth_mbs": 59.375
  },
  "results": [...]
}
```

## 🚨 Solución de Problemas

### Errores Comunes

**"No se puede conectar al servidor"**
- Verificar que `server_main.py` esté ejecutándose
- Verificar IP en `shared/config.py`
- Verificar firewall/puertos

**"Error inicializando cámara"**
- Verificar que la cámara no esté siendo usada por otra app
- Probar diferentes índices de cámara (0, 1, 2...)
- Verificar drivers de cámara

**"FPS muy bajo"**
- Reducir resolución o calidad JPEG
- Verificar uso de CPU/memoria
- Verificar latencia de red

### Comandos de Diagnóstico
```bash
# Verificar cámaras disponibles
python -c "from client.camera_handler import find_available_cameras; print(find_available_cameras())"

# Test de conectividad básica
python quick_test.py --profile connectivity

# Verificar servidor
curl http://192.168.1.84:5000/stats
```

## 📞 Próximos Pasos

Una vez completadas las pruebas de rendimiento:

1. **Analizar resultados** - Identificar configuración óptima
2. **Actualizar config.py** - Aplicar parámetros óptimos
3. **Integrar YOLOv8** - Añadir detección de objetos (Fase 2)
4. **Implementar Deep SORT** - Añadir tracking persistente (Fase 2)
5. **Optimizar pipeline** - Balancear detección vs throughput

¡Tu sistema está listo para validar la capacidad Full HD y proceder con el desarrollo del tracking de objetos!
