"""Pangu ONNX tensor order (not the order returned by the CDS service)."""

SURFACE = ("msl", "u10", "v10", "t2m")
UPPER = ("z", "q", "t", "u", "v")
LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
LATITUDES = tuple(90 - i * 0.25 for i in range(721))
LONGITUDES = tuple(i * 0.25 for i in range(1440))
STEPS = (24, 6, 3, 1)
UNITS = {"msl": "Pa", "u10": "m s-1", "v10": "m s-1", "t2m": "K",
         "z": "m2 s-2", "q": "kg kg-1", "t": "K", "u": "m s-1", "v": "m s-1"}
CDS_SURFACE = ("mean_sea_level_pressure", "10m_u_component_of_wind",
               "10m_v_component_of_wind", "2m_temperature")
CDS_UPPER = ("geopotential", "specific_humidity", "temperature",
             "u_component_of_wind", "v_component_of_wind")
