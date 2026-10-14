import math
from typing import Tuple, Dict, Any

class SpatialTileCoordinateTransformer:
    """
    Geographic bounding box and Web Mercator (EPSG:3857) tile coordinate converter
    for querying aerial imagery servers (e.g. OpenStreetMap, Sentinel-2, or Bavarian Geoportal).
    """
    @staticmethod
    def lat_lon_to_tile(lat: float, lon: float, zoom: int = 18) -> Tuple[int, int]:
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile

    @staticmethod
    def tile_to_bounding_box(xtile: int, ytile: int, zoom: int = 18) -> Dict[str, float]:
        n = 2.0 ** zoom
        lon_min = xtile / n * 360.0 - 180.0
        lat_rad_min = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_min = math.degrees(lat_rad_min)
        
        lon_max = (xtile + 1) / n * 360.0 - 180.0
        lat_rad_max = math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 1) / n)))
        lat_max = math.degrees(lat_rad_max)
        
        return {
            "north": round(lat_min, 6),
            "south": round(lat_max, 6),
            "west": round(lon_min, 6),
            "east": round(lon_max, 6)
        }
