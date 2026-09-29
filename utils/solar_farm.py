"""
Solar Farm simulation engine.
Generates realistic panel data, anomaly distributions, and alert feeds.
In production, this would connect to real SCADA/IoT sensor data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


DEFECT_TYPES = {
    'Hotspot': {'yield_loss': (15, 30), 'temp_delta': (12, 25), 'severity': 'critical'},
    'Micro-Crack': {'yield_loss': (5, 15), 'temp_delta': (3, 8), 'severity': 'warning'},
    'Dust Accumulation': {'yield_loss': (8, 20), 'temp_delta': (2, 5), 'severity': 'warning'},
    'Bird Dropping': {'yield_loss': (3, 10), 'temp_delta': (1, 4), 'severity': 'warning'},
    'Diode Bypass': {'yield_loss': (20, 40), 'temp_delta': (15, 30), 'severity': 'critical'},
    'PID Effect': {'yield_loss': (10, 25), 'temp_delta': (5, 12), 'severity': 'critical'},
    'Snail Trail': {'yield_loss': (3, 8), 'temp_delta': (1, 3), 'severity': 'warning'},
    'Delamination': {'yield_loss': (5, 15), 'temp_delta': (3, 7), 'severity': 'warning'},
}


class SolarFarm:
    def __init__(self, rows: int = 8, cols: int = 12, anomaly_rate: float = 0.12, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.rows = rows
        self.cols = cols
        self.total_panels = rows * cols
        self.anomaly_rate = anomaly_rate

        # Generate base efficiency grid (normal panels ~88-98%)
        self.base_efficiency = np.random.uniform(88, 98, (rows, cols))
        self.base_temperature = np.random.uniform(38, 52, (rows, cols))
        self.base_voltage = np.random.uniform(30, 42, (rows, cols))
        self.base_output = self.base_efficiency / 100 * np.random.uniform(0.28, 0.38, (rows, cols))

        # Generate anomalies
        n_anomalies = int(self.total_panels * anomaly_rate)
        anomaly_positions = random.sample(
            [(r, c) for r in range(rows) for c in range(cols)],
            n_anomalies
        )

        self.anomalous_panels = []
        self.panel_data = {}

        for r in range(rows):
            for c in range(cols):
                panel_id = f"PV-{r:02d}-{c:02d}"
                if (r, c) in anomaly_positions:
                    defect = random.choice(list(DEFECT_TYPES.keys()))
                    info = DEFECT_TYPES[defect]
                    yield_loss = random.uniform(*info['yield_loss'])
                    temp_delta = random.uniform(*info['temp_delta'])
                    efficiency = max(20, self.base_efficiency[r, c] - yield_loss)
                    temperature = self.base_temperature[r, c] + temp_delta
                    confidence = random.uniform(82, 99)

                    self.base_efficiency[r, c] = efficiency
                    self.base_temperature[r, c] = temperature

                    self.anomalous_panels.append({
                        'row': r,
                        'col': c,
                        'panel_id': panel_id,
                        'defect_type': defect,
                        'severity': info['severity'],
                        'yield_loss': yield_loss,
                        'temp_delta': temp_delta,
                        'confidence': confidence,
                    })

                    self.panel_data[(r, c)] = {
                        'panel_id': panel_id,
                        'efficiency': efficiency,
                        'temperature': temperature,
                        'voltage': self.base_voltage[r, c] * (efficiency / 100),
                        'output': self.base_output[r, c] * (efficiency / 100),
                        'status': 'Critical' if info['severity'] == 'critical' else 'Degraded',
                        'defect_type': defect,
                        'confidence': confidence,
                        'yield_loss': yield_loss,
                    }
                else:
                    self.panel_data[(r, c)] = {
                        'panel_id': panel_id,
                        'efficiency': self.base_efficiency[r, c],
                        'temperature': self.base_temperature[r, c],
                        'voltage': self.base_voltage[r, c],
                        'output': self.base_output[r, c],
                        'status': 'Healthy',
                        'defect_type': 'None',
                        'confidence': 0,
                        'yield_loss': 0,
                    }

    def get_grid_heatmap(self) -> np.ndarray:
        return self.base_efficiency

    def get_panel_statuses(self) -> dict:
        statuses = {}
        for (r, c), data in self.panel_data.items():
            statuses[(r, c)] = data['status']
        return statuses

    def get_panel_detail(self, row: int, col: int) -> dict:
        return self.panel_data.get((row, col), {})

    def get_kpi_summary(self) -> dict:
        efficiencies = [d['efficiency'] for d in self.panel_data.values()]
        healthy = sum(1 for d in self.panel_data.values() if d['status'] == 'Healthy')
        critical = sum(1 for p in self.anomalous_panels if p['severity'] == 'critical')
        warning = sum(1 for p in self.anomalous_panels if p['severity'] == 'warning')
        total_output = sum(d['output'] for d in self.panel_data.values())

        return {
            'production': total_output * 24 * 0.65,  # kWh estimate
            'prod_trend': random.uniform(2.5, 8.3),
            'health_score': np.mean(efficiencies),
            'health_trend': random.uniform(-1.5, 2.0),
            'healthy_panels': healthy,
            'total_panels': self.total_panels,
            'anomaly_count': len(self.anomalous_panels),
            'critical_count': critical,
            'warning_count': warning,
        }

    def get_alerts(self) -> list:
        alerts = []
        for panel in sorted(self.anomalous_panels, key=lambda x: x['severity'] == 'critical', reverse=True):
            desc_map = {
                'Hotspot': f"Panel temperature {self.panel_data[(panel['row'], panel['col'])]['temperature']:.0f}°C — significantly above nominal. Likely cell interconnection degradation.",
                'Micro-Crack': "Hairline fracture detected. Output reduced but stable. Schedule inspection.",
                'Dust Accumulation': "Surface soiling reducing light absorption. Cleaning recommended.",
                'Bird Dropping': "Localized shading from organic debris. Minor output impact.",
                'Diode Bypass': "Bypass diode activated — entire substring offline. Immediate repair needed.",
                'PID Effect': "Potential-Induced Degradation detected. Voltage leakage to frame.",
                'Snail Trail': "Cosmetic snail trail pattern. Minor efficiency impact.",
                'Delamination': "Encapsulant separation detected. Moisture ingress risk.",
            }
            alerts.append({
                'title': f"{panel['defect_type']} — {panel['panel_id']}",
                'description': desc_map.get(panel['defect_type'], "Anomaly detected."),
                'severity': panel['severity'],
                'panel_id': panel['panel_id'],
                'yield_loss': panel['yield_loss'],
            })
        return alerts

    def get_string_performance(self) -> dict:
        strings = {}
        for r in range(self.rows):
            name = f"String {r + 1}"
            row_eff = [self.panel_data[(r, c)]['efficiency'] for c in range(self.cols)]
            strings[name] = np.mean(row_eff)
        return dict(sorted(strings.items(), key=lambda x: x[1], reverse=True))

    def get_defect_distribution(self) -> dict:
        dist = {}
        for panel in self.anomalous_panels:
            dt = panel['defect_type']
            dist[dt] = dist.get(dt, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    def get_temp_efficiency_scatter(self) -> pd.DataFrame:
        rows = []
        for (r, c), data in self.panel_data.items():
            rows.append({
                'panel_id': data['panel_id'],
                'temperature': data['temperature'],
                'efficiency': data['efficiency'],
                'status': data['status'],
            })
        return pd.DataFrame(rows)
