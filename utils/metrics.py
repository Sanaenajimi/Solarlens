"""
Metrics Calculator.
Generates realistic solar production curves, historical trends, and yield loss estimates.
"""

import numpy as np
import pandas as pd
from datetime import datetime


class MetricsCalculator:
    def generate_solar_curve(self, hours: pd.DatetimeIndex) -> dict:
        """Generate a realistic bell-shaped solar production curve."""
        n = len(hours)
        actual = []
        expected = []

        for i, h in enumerate(hours):
            hour = h.hour + h.minute / 60

            # Bell curve peaking at 13:00
            peak = 42  # kW peak
            solar = peak * np.exp(-0.5 * ((hour - 13) / 3.2) ** 2)

            # Night = 0
            if hour < 6 or hour > 20.5:
                solar = 0

            # Cloud transients
            cloud_factor = 1.0 - 0.15 * np.sin(i * 0.7 + 2.1) * max(0, np.sin(i * 0.3))
            noise = np.random.normal(0, 0.8)

            actual_val = max(0, solar * cloud_factor + noise)
            expected_val = max(0, solar * 1.0)

            actual.append(actual_val)
            expected.append(expected_val)

        return {'actual': actual, 'expected': expected}

    def generate_panel_history(self, panel_info: dict, dates: pd.DatetimeIndex) -> dict:
        """Generate 30-day history for a specific panel."""
        base_eff = panel_info.get('efficiency', 92)
        base_temp = panel_info.get('temperature', 45)
        n = len(dates)

        # Gradual degradation for anomalous panels
        if panel_info.get('defect_type', 'None') != 'None':
            degradation = np.linspace(base_eff + 8, base_eff, n)
            temp_rise = np.linspace(base_temp - 5, base_temp, n)
        else:
            degradation = np.full(n, base_eff)
            temp_rise = np.full(n, base_temp)

        efficiency = degradation + np.random.normal(0, 1.5, n)
        temperature = temp_rise + np.random.normal(0, 2, n)

        # Seasonal variation
        for i in range(n):
            day_of_year = dates[i].timetuple().tm_yday
            seasonal = 3 * np.sin(2 * np.pi * day_of_year / 365)
            temperature[i] += seasonal

        return {
            'efficiency': np.clip(efficiency, 30, 100),
            'temperature': np.clip(temperature, 15, 85),
        }

    def generate_monthly_trend(self, months: pd.DatetimeIndex) -> dict:
        """Generate monthly anomaly detection/resolution trend."""
        n = len(months)
        detected = []
        resolved = []

        for i in range(n):
            base = int(np.random.poisson(8))
            detected.append(base + int(np.random.uniform(0, 4)))
            resolved.append(max(0, base - int(np.random.uniform(0, 3))))

        return {'detected': detected, 'resolved': resolved}

    def get_yield_loss_table(self) -> pd.DataFrame:
        """Estimated financial impact by defect type."""
        data = {
            'Defect Type': ['Hotspot', 'Diode Bypass', 'PID Effect', 'Dust Accumulation', 'Micro-Crack', 'Bird Dropping', 'Snail Trail', 'Delamination'],
            'Panels Affected': [3, 2, 2, 4, 3, 2, 1, 1],
            'Yield Loss (%)': [22.5, 32.0, 17.5, 14.0, 10.0, 6.5, 5.0, 10.0],
            'Revenue Impact (€/yr)': [4725, 8960, 4900, 3920, 2100, 910, 350, 700],
            'Priority': ['🔴 Critical', '🔴 Critical', '🔴 Critical', '🟡 Medium', '🟡 Medium', '🟢 Low', '🟢 Low', '🟡 Medium'],
        }
        return pd.DataFrame(data)
