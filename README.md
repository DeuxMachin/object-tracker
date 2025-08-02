# Sistema de Seguimiento Multi-Objeto en Tiempo Real

## Descripción del Proyecto

Este sistema implementa un framework completo de seguimiento multi-objeto (Multiple Object Tracking - MOT) que combina técnicas avanzadas de visión computacional y aprendizaje profundo para rastrear múltiples entidades en secuencias de video en tiempo real.

El sistema integra detección de objetos basada en redes neuronales convolucionales, algoritmos de asociación de datos probabilísticos y filtros de estado para mantener consistencia temporal en el seguimiento, proporcionando identificadores únicos persistentes para cada objeto detectado a lo largo de la secuencia.

## Arquitectura del Sistema

### Pipeline de Procesamiento

El sistema sigue una arquitectura modular de múltiples etapas:

1. **Adquisición de Video**: Captura de frames desde dispositivos de entrada (webcam, archivos de video)
2. **Preprocesamiento**: Normalización, redimensionamiento y augmentación de datos
3. **Detección de Objetos**: Identificación y localización de entidades mediante redes neuronales
4. **Extracción de Características**: Generación de descriptores visuales para asociación
5. **Asociación de Datos**: Emparejamiento temporal usando algoritmos probabilísticos
6. **Estimación de Estado**: Predicción de trayectorias mediante filtros estadísticos
7. **Post-procesamiento**: Refinamiento de resultados y gestión de identificadores

### Aplicaciones

- **Sistemas de Videovigilancia**: Análisis comportamental y detección de anomalías
- **Análisis de Tráfico Vehicular**: Conteo, clasificación y análisis de flujo vehicular
- **Monitoreo Deportivo**: Tracking de jugadores y análisis de rendimiento
- **Investigación en Visión Computacional**: Evaluación de algoritmos MOT
- **Robótica Móvil**: Navegación y evasión de obstáculos dinámicos

## Fundamentos Teóricos

### Detección de Objetos

El sistema utiliza **YOLOv8** (You Only Look Once version 8), una red neuronal convolucional de una sola etapa que reformula la detección de objetos como un problema de regresión. A diferencia de los detectores de dos etapas (como R-CNN), YOLO divide la imagen de entrada en una grilla S×S y predice simultáneamente:

- **Coordenadas de bounding boxes**: (x, y, w, h) normalizadas
- **Confianza de objetividad**: P(Object) * IoU(pred, truth)
- **Probabilidades de clase**: P(Class_i | Object)

La función de pérdida combina errores de localización, confianza y clasificación:

```
Loss = λ_coord * L_coord + λ_obj * L_obj + λ_noobj * L_noobj + λ_class * L_class
```

### Seguimiento Multi-Objeto (MOT)

#### Deep SORT Algorithm

El algoritmo **Deep SORT** extiende el framework SORT (Simple Online and Realtime Tracking) integrando:

1. **Filtro de Kalman**: Para estimación de estado y predicción de movimiento
2. **Algoritmo Húngaro**: Para asociación óptima de detecciones-tracks
3. **Deep Association Metric**: CNN para extracción de características visuales

#### Modelo de Estado

El filtro de Kalman modela el estado de cada objeto como un vector 8-dimensional:

```
x = [u, v, s, r, u̇, v̇, ṡ, ṙ]ᵀ
```

Donde:
- (u, v): Centro del bounding box
- s: Escala (área)
- r: Aspect ratio
- Las derivadas representan velocidades en el espacio de imagen

#### Matrices del Sistema

**Matriz de Transición F** (movimiento uniforme):
```
F = [I_4x4  I_4x4]
    [0_4x4  I_4x4]
```

**Matriz de Observación H** (observamos solo posición y escala):
```
H = [I_4x4  0_4x4]
```

### Asociación de Datos

#### Distancia de Mahalanobis

Para asociación basada en movimiento:
```
d(1)(i,j) = (d_j - y_i)ᵀ S_i⁻¹ (d_j - y_i)
```

Donde:
- d_j: j-ésima detección
- y_i: predicción del i-ésimo track
- S_i: matriz de covarianza de innovación

#### Distancia Coseno

Para asociación basada en apariencia usando características CNN:
```
d(2)(i,j) = min{1 - r_j^T r_k^(i) | r_k^(i) ∈ R_i}
```

Donde R_i contiene las últimas L_k características confirmadas del track i.

#### Métrica Combinada

```
c_{i,j} = λ d(1)(i,j) + (1-λ) d(2)(i,j)
```

### Stack Tecnológico

#### Componentes Core

- **Python 3.8+**: Lenguaje de desarrollo principal
- **OpenCV 4.8+**: Biblioteca de visión computacional para procesamiento de imágenes
- **NumPy**: Computación numérica y álgebra lineal optimizada
- **PyTorch**: Framework de deep learning para inferencia de modelos

#### Detección y Tracking

- **Ultralytics YOLOv8**: Detector de objetos pre-entrenado en COCO dataset
- **Deep SORT**: Implementación de tracking multi-objeto
- **SciPy**: Algoritmos de optimización (Algoritmo Húngaro)

#### Interfaz y Visualización

- **Flask**: Framework web para API REST y interfaz de usuario
- **WebRTC**: Streaming de video en tiempo real
- **Matplotlib/Plotly**: Visualización de trayectorias y métricas

#### Persistencia de Datos

- **Supabase**: Base de datos PostgreSQL como servicio
- **SQLAlchemy**: ORM para manejo de base de datos
- **Redis**: Cache para optimización de rendimiento

## Roadmap de Desarrollo

### Fase 1: Infraestructura Base (Completado)
- [x] Configuración del entorno de desarrollo
- [x] Implementación del módulo de captura de video
- [x] Arquitectura modular del sistema
- [x] Manejo de múltiples backends de cámara (V4L2, GStreamer, FFMPEG)
- [x] Sistema de configuración centralizado

### Fase 2: Integración de Detección
- [ ] Implementación del pipeline YOLOv8
- [ ] Optimización de inferencia (batch processing, TensorRT)
- [ ] Filtrado post-procesamiento (NMS, confidence thresholding)
- [ ] Sistema de clases configurables (COCO, custom datasets)

### Fase 3: Motor de Tracking
- [ ] Implementación del algoritmo Deep SORT
- [ ] Integración del filtro de Kalman para predicción de estado
- [ ] Sistema de extracción de características visuales (CNN features)
- [ ] Algoritmo de asociación Hungarian con métricas combinadas
- [ ] Gestión de ciclo de vida de tracks (birth, update, death)

### Fase 4: Interfaz Web y API
- [ ] Desarrollo de API RESTful con Flask
- [ ] Streaming de video en tiempo real (WebRTC/WebSocket)
- [ ] Dashboard de monitoreo con métricas en vivo
- [ ] Sistema de configuración web para parámetros del tracker

### Fase 5: Persistencia y Analytics
- [ ] Integración con base de datos Supabase
- [ ] Esquema de datos para trayectorias y eventos
- [ ] Sistema de analytics y generación de reportes
- [ ] Exportación de datos (JSON, CSV, formato MOT Challenge)

## Instalación y Configuración

### Requisitos del Sistema

**Hardware Mínimo:**
- CPU: Intel Core i5 8th gen / AMD Ryzen 5 2600 o superior
- RAM: 8GB DDR4 (16GB recomendado para procesamiento en tiempo real)
- GPU: NVIDIA GTX 1060 / RTX 2060 o superior (opcional, para aceleración CUDA)
- Almacenamiento: 10GB de espacio libre

**Software:**
- Python 3.8+ con pip
- CUDA Toolkit 11.8+ (para aceleración GPU)
- Git para control de versiones

### Configuración del Entorno

```bash
# Clonar el repositorio
git clone https://github.com/DeuxMachin/object-tracker.git
cd object-tracker

# Crear entorno virtual
python -m venv SeguimientoObjetos_env
source SeguimientoObjetos_env/bin/activate  # Linux/macOS
# o
SeguimientoObjetos_env\Scripts\activate     # Windows

# Instalar dependencias base
pip install -r requirements.txt

# Verificar instalación
python main.py
```

### Configuración Avanzada

**Optimización GPU (NVIDIA):**
```bash
# Verificar disponibilidad CUDA
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"

# Instalar PyTorch con soporte CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Configuración de Parámetros:**

Editar `src/utils/config.py`:

```python
# Configuración de cámara
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Parámetros YOLOv8
DETECTION_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.4
MODEL_SIZE = 'yolov8n'  # n, s, m, l, x

# Parámetros Deep SORT
MAX_DISAPPEARED = 30
MAX_DISTANCE = 0.7
NN_BUDGET = 100
```

## Métricas y Evaluación

### Métricas de Tracking

**MOTA (Multiple Object Tracking Accuracy):**
```
MOTA = 1 - (FN + FP + IDSW) / GT
```

**MOTP (Multiple Object Tracking Precision):**
```
MOTP = Σ d_t / Σ c_t
```

**IDF1 (ID F1 Score):**
```
IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
```

### Benchmarking

El sistema soporta evaluación contra datasets estándar:
- **MOT Challenge**: MOT15, MOT16, MOT17, MOT20
- **KITTI Tracking**: Secuencias de tráfico vehicular
- **Custom Datasets**: Formato personalizable

## Troubleshooting

### Problemas Comunes de Configuración

**Error: "No se detecta cámara"**
```bash
# Verificar dispositivos disponibles
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"

# Probar diferentes backends
export OPENCV_VIDEOIO_PRIORITY_MSMF=0  # Windows
export OPENCV_VIDEOIO_PRIORITY_V4L2=0  # Linux
```

**Error: "ModuleNotFoundError"**
```bash
# Verificar entorno virtual activo
which python  # debe apuntar al venv

# Reinstalar dependencias
pip install --upgrade --force-reinstall -r requirements.txt
```

**Performance Issues:**
- Reducir resolución de cámara en `config.py`
- Usar modelo YOLOv8n en lugar de versiones más grandes
- Habilitar aceleración GPU si está disponible

### Consideraciones de Rendimiento

**Optimizaciones CPU:**
- Compilar OpenCV con optimizaciones AVX/SSE
- Usar threading para paralelizar detección y tracking
- Implementar frame skipping adaptativo

**Optimizaciones GPU:**
- Utilizar TensorRT para optimización de modelos
- Batch processing para múltiples detecciones
- Memory pooling para reducir allocaciones

## Contribución y Desarrollo


### Guidelines de Desarrollo

1. **Documentación**: Todas las funciones deben incluir docstrings con descripción de parámetros y valores de retorno
2. **Testing**: Implementar unit tests para componentes críticos
3. **Performance**: Perfilar código con `cProfile` antes de optimizaciones
4. **Reproducibilidad**: Fijar seeds aleatorias para experimentos consistentes

### Referencias Académicas

- Bewley, A., et al. "Simple online and realtime tracking." ICIP 2016
- Wojke, N., et al. "Simple online and realtime tracking with a deep association metric." ICIP 2017
- Jocher, G., et al. "YOLOv8: A new state-of-the-art for object detection." 2023
- Kalman, R. E. "A new approach to linear filtering and prediction problems." 1960

---

**Licencia**: MIT License  
**Autor**: Edward Contreras (DeuxMachine)  
**Repositorio**: https://github.com/DeuxMachin/object-tracker