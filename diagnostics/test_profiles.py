"""
Configuraciones de Prueba de Rendimiento

Define configuraciones optimizadas para diferentes escenarios de uso.
Permite personalizar fácilmente los parámetros según las capacidades del hardware.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class TestProfile:
    """Perfil de prueba con configuración específica."""
    name: str
    description: str
    width: int
    height: int
    fps: int
    jpeg_quality: int
    duration_seconds: int = 60
    warmup_seconds: int = 10
    
    @property
    def resolution_name(self) -> str:
        """Nombre amigable de la resolución."""
        if self.width == 640 and self.height == 480:
            return "VGA"
        elif self.width == 1280 and self.height == 720:
            return "HD 720p"
        elif self.width == 1920 and self.height == 1080:
            return "Full HD 1080p"
        elif self.width == 2560 and self.height == 1440:
            return "QHD 1440p"
        elif self.width == 3840 and self.height == 2160:
            return "4K UHD"
        else:
            return f"{self.width}x{self.height}"
    
    @property
    def estimated_raw_mbps(self) -> float:
        """Estima los MB/s necesarios sin compresión."""
        # 3 bytes por pixel (RGB), convertido a MB/s
        bytes_per_frame = self.width * self.height * 3
        bytes_per_second = bytes_per_frame * self.fps
        return bytes_per_second / (1024 * 1024)
    
    @property
    def estimated_compressed_mbps(self) -> float:
        """Estima los MB/s con compresión JPEG."""
        # Factor de compresión aproximado basado en calidad JPEG
        compression_factor = {
            90: 0.15,  # Alta calidad, poca compresión
            80: 0.10,  # Calidad media-alta
            70: 0.08,  # Calidad media
            60: 0.06,  # Calidad media-baja
            50: 0.04,  # Baja calidad, alta compresión
        }
        factor = compression_factor.get(self.jpeg_quality, 0.10)
        return self.estimated_raw_mbps * factor

# Perfiles de prueba predefinidos
PERFORMANCE_PROFILES: List[TestProfile] = [
    
    # Pruebas básicas de conectividad
    TestProfile(
        name="connectivity_test",
        description="Prueba básica de conectividad y latencia",
        width=320,
        height=240,
        fps=10,
        jpeg_quality=50,
        duration_seconds=30,
        warmup_seconds=5
    ),
    
    # Calidad estándar para equipos limitados
    TestProfile(
        name="standard_quality",
        description="Calidad estándar VGA para equipos con recursos limitados",
        width=640,
        height=480,
        fps=15,
        jpeg_quality=70,
        duration_seconds=60,
        warmup_seconds=10
    ),
    
    # HD para streaming regular
    TestProfile(
        name="hd_streaming",
        description="HD 720p para streaming regular",
        width=1280,
        height=720,
        fps=20,
        jpeg_quality=75,
        duration_seconds=90,
        warmup_seconds=15
    ),
    
    # Full HD conservador
    TestProfile(
        name="full_hd_conservative",
        description="Full HD 1080p con configuración conservadora",
        width=1920,
        height=1080,
        fps=15,
        jpeg_quality=70,
        duration_seconds=120,
        warmup_seconds=20
    ),
    
    # Full HD balanceado
    TestProfile(
        name="full_hd_balanced",
        description="Full HD 1080p balanceado para la mayoría de casos",
        width=1920,
        height=1080,
        fps=20,
        jpeg_quality=75,
        duration_seconds=120,
        warmup_seconds=20
    ),
    
    # Full HD óptimo
    TestProfile(
        name="full_hd_optimal",
        description="Full HD 1080p con configuración óptima",
        width=1920,
        height=1080,
        fps=25,
        jpeg_quality=80,
        duration_seconds=120,
        warmup_seconds=20
    ),
    
    # Full HD máximo rendimiento
    TestProfile(
        name="full_hd_maximum",
        description="Full HD 1080p a máximo rendimiento (30 FPS)",
        width=1920,
        height=1080,
        fps=30,
        jpeg_quality=85,
        duration_seconds=180,
        warmup_seconds=30
    ),
    
    # Prueba de calidad máxima
    TestProfile(
        name="maximum_quality",
        description="Máxima calidad de imagen (baja compresión)",
        width=1920,
        height=1080,
        fps=30,
        jpeg_quality=95,
        duration_seconds=60,
        warmup_seconds=15
    ),
    
    # Prueba de estrés
    TestProfile(
        name="stress_test",
        description="Prueba de estrés con alta demanda",
        width=1920,
        height=1080,
        fps=60,
        jpeg_quality=90,
        duration_seconds=300,  # 5 minutos
        warmup_seconds=30
    ),
]

# Configuraciones para diferentes tipos de red
NETWORK_PROFILES = {
    "lan_gigabit": {
        "name": "LAN Gigabit",
        "description": "Red LAN Gigabit (1000 Mbps)",
        "max_mbps": 125,  # 1000 Mbps / 8
        "recommended_profiles": [
            "full_hd_conservative",
            "full_hd_balanced", 
            "full_hd_optimal",
            "full_hd_maximum"
        ]
    },
    
    "lan_fast": {
        "name": "LAN Fast Ethernet",
        "description": "Red LAN Fast Ethernet (100 Mbps)",
        "max_mbps": 12.5,  # 100 Mbps / 8
        "recommended_profiles": [
            "standard_quality",
            "hd_streaming",
            "full_hd_conservative"
        ]
    },
    
    "wifi_ac": {
        "name": "WiFi 802.11ac",
        "description": "WiFi AC (hasta 500 Mbps real)",
        "max_mbps": 60,  # Considerando overhead WiFi
        "recommended_profiles": [
            "hd_streaming",
            "full_hd_conservative",
            "full_hd_balanced"
        ]
    },
    
    "wifi_n": {
        "name": "WiFi 802.11n",
        "description": "WiFi N (hasta 150 Mbps real)",
        "max_mbps": 18,
        "recommended_profiles": [
            "standard_quality",
            "hd_streaming"
        ]
    }
}

def get_profile_by_name(name: str) -> TestProfile:
    """Obtiene un perfil por su nombre."""
    for profile in PERFORMANCE_PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"Perfil '{name}' no encontrado")

def get_profiles_for_network(network_type: str) -> List[TestProfile]:
    """Obtiene perfiles recomendados para un tipo de red."""
    if network_type not in NETWORK_PROFILES:
        raise ValueError(f"Tipo de red '{network_type}' no válido")
    
    network = NETWORK_PROFILES[network_type]
    profiles = []
    
    for profile_name in network["recommended_profiles"]:
        try:
            profiles.append(get_profile_by_name(profile_name))
        except ValueError:
            continue
    
    return profiles

def print_all_profiles():
    """Imprime todos los perfiles disponibles con información detallada."""
    print("📋 PERFILES DE PRUEBA DISPONIBLES")
    print("=" * 80)
    
    for profile in PERFORMANCE_PROFILES:
        print(f"\n🎯 {profile.name}")
        print(f"   📝 {profile.description}")
        print(f"   📐 Resolución: {profile.resolution_name} ({profile.width}x{profile.height})")
        print(f"   🎬 FPS: {profile.fps}")
        print(f"   🗜️ Calidad JPEG: {profile.jpeg_quality}%")
        print(f"   ⏱️ Duración: {profile.duration_seconds}s (warmup: {profile.warmup_seconds}s)")
        print(f"   📊 Estimado sin compresión: {profile.estimated_raw_mbps:.1f} MB/s")
        print(f"   📦 Estimado con compresión: {profile.estimated_compressed_mbps:.1f} MB/s")

def print_network_recommendations():
    """Imprime recomendaciones por tipo de red."""
    print("\n🌐 RECOMENDACIONES POR TIPO DE RED")
    print("=" * 60)
    
    for net_type, net_info in NETWORK_PROFILES.items():
        print(f"\n📡 {net_info['name']}")
        print(f"   📝 {net_info['description']}")
        print(f"   🚀 Ancho de banda máximo: {net_info['max_mbps']} MB/s")
        print(f"   ✅ Perfiles recomendados:")
        
        for profile_name in net_info["recommended_profiles"]:
            try:
                profile = get_profile_by_name(profile_name)
                print(f"      - {profile.name}: {profile.resolution_name} @ {profile.fps}fps")
            except ValueError:
                continue

if __name__ == "__main__":
    print_all_profiles()
    print_network_recommendations()
