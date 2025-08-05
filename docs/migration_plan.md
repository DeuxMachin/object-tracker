# Plan de Migración: Local → Remoto

## Componentes que SE MANTIENEN:
- ✅ `VideoStreamer.capture_loop()` - Loop de captura
- ✅ `VideoStreamer.compress_frame()` - Compresión JPEG
- ✅ `VideoStreamer.get_frame()` - Obtener frame actual
- ✅ `CameraHandler` - Manejo de cámara
- ✅ Control de FPS y estadísticas

## Componentes que SE REEMPLAZAN:

### ACTUAL (Testing Local):
```python
# Flask server para visualización local
app = Flask(__name__)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames())

# Función de testing
def test_local_streaming():
    app.run(host='0.0.0.0', port=5000)
```

### FUTURO (Conexión Remota):
```python
# Network sender para transmisión remota
class NetworkSender:
    def send_frame(self, frame_data):
        # Enviar por WebSocket/TCP/UDP
        pass

# Función de producción
def start_remote_streaming(server_ip, server_port):
    sender = NetworkSender(server_ip, server_port)
    # Usar mismo VideoStreamer pero enviar frames por red
```

## Modificaciones Necesarias:

### 1. Agregar NetworkSender:
- Conexión WebSocket/TCP al servidor
- Manejo de reconexión automática
- Buffer para frames perdidos
- Compresión adicional si es necesario

### 2. Modificar capture_loop():
```python
# ANTES (Local):
with self.frame_lock:
    self.current_frame = compressed_frame

# DESPUÉS (Remoto):
with self.frame_lock:
    self.current_frame = compressed_frame
    
# NUEVO: Enviar al servidor
if self.network_sender:
    self.network_sender.send_frame(compressed_frame)
```

### 3. Configuración Dual:
```python
class StreamConfig:
    MODE_LOCAL = 'local'    # Flask server
    MODE_REMOTE = 'remote'  # Network transmission
    
    def __init__(self, mode=MODE_LOCAL):
        self.mode = mode
        if mode == MODE_REMOTE:
            self.server_ip = "192.168.1.100"
            self.server_port = 8080
```

## Protocolo de Transmisión:

### Opción 1: WebSocket (RECOMENDADO)
- ✅ Bidireccional (servidor puede enviar comandos)
- ✅ Reconexión automática
- ✅ Compatible con web browsers
- ✅ Headers HTTP estándar

### Opción 2: TCP Raw
- ✅ Máximo rendimiento
- ✅ Garantía de entrega
- ❌ Sin reconexión automática
- ❌ Manejo manual de protocolo

### Opción 3: UDP + Custom Protocol
- ✅ Menor latencia
- ✅ No bloquea en pérdidas
- ❌ Sin garantía de entrega
- ❌ Más complejo de implementar

## Timeline de Migración:

### Semana 1: Preparación
- [ ] Crear NetworkSender clase base
- [ ] Implementar WebSocket client
- [ ] Agregar configuración dual (local/remote)

### Semana 2: Integración
- [ ] Modificar VideoStreamer para soportar ambos modos
- [ ] Testing con servidor dummy
- [ ] Optimizar buffer y reconexión

### Semana 3: Deployment
- [ ] Setup servidor en Equipo B
- [ ] Configurar red entre equipos
- [ ] Testing completo end-to-end
- [ ] Monitoreo y ajuste de rendimiento
