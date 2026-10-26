"""Parses VCH Framework.xlsx into numpy arrays for the simulation engine."""
import os
import numpy as np
import openpyxl

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(APP_DIR, "VCH Framework.xlsx")

FLUIDS = {}
METALS = {}
INSULATION = {}
IMPELLERS = []

def _load_excel():
    global FLUIDS, METALS, INSULATION, IMPELLERS
    if not os.path.exists(EXCEL_PATH):
        print(f"WARNING: {EXCEL_PATH} not found. Using fallback materials.")
        _load_fallback()
        return

    try:
        wb = openpyxl.load_workbook(EXCEL_PATH, data_only=True)
        
        # --- 1. FLUIDS ---
        if "Fluids" in wb.sheetnames:
            ws = wb["Fluids"]
            current_name = None
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0]: current_name = str(row[0]).strip()
                if row[1] is not None and current_name:
                    if current_name not in FLUIDS: FLUIDS[current_name] = []
                    t = float(row[1])
                    rho = float(row[3]) if row[3] else 1000.0
                    cp = float(row[4]) if row[4] else 4000.0
                    mu = float(row[5]) if isinstance(row[5], (int, float)) else 1.0
                    k = float(row[6]) if row[6] else 0.5
                    dH = 0.0
                    if row[7] and isinstance(row[7], str) and "(" in row[7]:
                        try: dH = float(row[7].split("(")[0].strip())
                        except: pass
                    elif isinstance(row[7], (int, float)):
                        dH = float(row[7])
                    FLUIDS[current_name].append((t, rho, cp, mu, k, dH))
            
            for name in FLUIDS:
                FLUIDS[name] = np.array(sorted(FLUIDS[name], key=lambda x: x[0]))

        # --- 2. METALS ---
        if "Metals" in wb.sheetnames:
            ws = wb["Metals"]
            current_name = None
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[1]: current_name = str(row[1]).strip()
                if row[2] is not None and current_name:
                    if current_name not in METALS: METALS[current_name] = []
                    t = float(row[2])
                    rho = float(row[3]) if row[3] else 7800.0
                    cp = float(row[4]) if row[4] else 500.0
                    k = float(row[5]) if row[5] else 15.0
                    METALS[current_name].append((t, rho, cp, k))
            
            for name in METALS:
                METALS[name] = np.array(sorted(METALS[name], key=lambda x: x[0]))

        # --- 3. INSULATION ---
        if "Insulation" in wb.sheetnames:
            ws = wb["Insulation"]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0] and row[1]:
                    INSULATION[str(row[0]).strip()] = float(row[1])

        # --- 4. IMPELLERS ---
        if "Impellers" in wb.sheetnames:
            ws = wb["Impellers"]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0] and row[2] and row[3]:
                    IMPELLERS.append(dict(
                        name=str(row[0]).strip(),
                        pattern=str(row[1]).strip() if row[1] else "",
                        C=float(row[2]),
                        a=float(row[3])
                    ))
    except Exception as e:
        print(f"Error reading Excel: {e}. Falling back.")
        _load_fallback()

def _load_fallback():
    global FLUIDS, METALS, INSULATION, IMPELLERS
    FLUIDS = {
        "Water": np.array([(0, 999.8, 4217, 0.001792, 0.561, 0), (100, 958.4, 4216, 0.000282, 0.679, 0)]),
        "RBD Palm Oil": np.array([(0, 925.0, 1920.0, 1.0, 0.180, 8.0), (150, 821.5, 2270.0, 0.002469, 0.14, 0)])
    }
    METALS = {
        "Stainless Steel": np.array([(0, 7985, 465, 13.5), (200, 7955, 535, 18.1)]),
        "Mild Steel": np.array([(0, 7850, 432, 52.2), (200, 7850, 516, 44.6)]),
        "Copper": np.array([(0, 8960, 377, 403), (200, 8910, 421, 379)])
    }
    INSULATION = {"Mineral Wool (Rockwool)": 0.038}
    IMPELLERS = [dict(name="Flat-Blade Turbine (6-Blade) Radial", pattern="Radial", C=0.74, a=0.67)]

_load_excel()