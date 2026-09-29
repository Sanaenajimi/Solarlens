"""
Anomaly Detection Engine.
Generates simulated thermal maps and defect probability distributions.
In production, this wraps the trained ResNet-50 + SE model for inference.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class AnomalyDetectionEngine:
    """Simulates the AI inference pipeline for the Streamlit demo."""

    DEFECT_CLASSES = [
        'Clean', 'Dusty', 'Snow-Covered', 'Bird Dropping',
        'Electrical Fault', 'Physical Damage', 'Hotspot',
        'Micro-Crack', 'PID Effect'
    ]

    def generate_thermal_map(self, panel_info: dict, size: int = 24) -> np.ndarray:
        """
        Generate a simulated thermal map for a panel.
        Healthy panels: uniform temperature.
        Anomalous panels: localized hotspots.
        """
        base_temp = panel_info.get('temperature', 45)
        thermal = np.random.normal(base_temp - 5, 1.5, (size, size))

        if panel_info.get('defect_type', 'None') != 'None':
            defect = panel_info['defect_type']

            if defect in ('Hotspot', 'Diode Bypass', 'PID Effect'):
                # Concentrated hot region
                n_spots = np.random.randint(1, 4)
                for _ in range(n_spots):
                    cx, cy = np.random.randint(4, size - 4, 2)
                    spot_radius = np.random.uniform(2, 5)
                    for i in range(size):
                        for j in range(size):
                            dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                            if dist < spot_radius:
                                thermal[i, j] += (spot_radius - dist) * np.random.uniform(3, 6)

            elif defect in ('Dust Accumulation', 'Bird Dropping'):
                # Gradient pattern
                gradient = np.linspace(0, np.random.uniform(3, 8), size)
                angle = np.random.uniform(0, np.pi)
                for i in range(size):
                    for j in range(size):
                        pos = i * np.cos(angle) + j * np.sin(angle)
                        thermal[i, j] += gradient[int(pos * (size - 1) / (size * 1.4)) % size]

            elif defect == 'Micro-Crack':
                # Line pattern
                direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
                if direction == 'horizontal':
                    row = np.random.randint(3, size - 3)
                    thermal[row - 1:row + 2, :] += np.random.uniform(4, 10)
                elif direction == 'vertical':
                    col = np.random.randint(3, size - 3)
                    thermal[:, col - 1:col + 2] += np.random.uniform(4, 10)
                else:
                    for k in range(size):
                        offset = int(k * 0.8)
                        if 0 <= offset < size:
                            thermal[k, max(0, offset - 1):min(size, offset + 2)] += np.random.uniform(4, 8)

            elif defect in ('Snail Trail', 'Delamination'):
                # Distributed noise
                noise = np.random.uniform(0, 5, (size, size))
                thermal += gaussian_filter(noise, sigma=3)

        # Smooth the thermal map
        thermal = gaussian_filter(thermal, sigma=1.2)
        return thermal

    def get_defect_probabilities(self, panel_info: dict) -> dict:
        """
        Generate a probability distribution over defect classes.
        The top class matches the panel's assigned defect type.
        """
        defect = panel_info.get('defect_type', 'None')
        confidence = panel_info.get('confidence', 95)

        # Map internal defect types to classification classes
        class_map = {
            'Hotspot': 'Electrical Fault',
            'Micro-Crack': 'Physical Damage',
            'Dust Accumulation': 'Dusty',
            'Bird Dropping': 'Bird Dropping',
            'Diode Bypass': 'Electrical Fault',
            'PID Effect': 'Electrical Fault',
            'Snail Trail': 'Physical Damage',
            'Delamination': 'Physical Damage',
            'None': 'Clean',
        }

        top_class = class_map.get(defect, 'Clean')
        classes = ['Clean', 'Dusty', 'Snow-Covered', 'Bird Dropping', 'Electrical Fault', 'Physical Damage']

        probs = {}
        remaining = 100 - confidence
        for cls in classes:
            if cls == top_class:
                probs[cls] = confidence
            else:
                probs[cls] = max(0.1, remaining / (len(classes) - 1) + np.random.uniform(-2, 2))

        # Normalize
        total = sum(probs.values())
        probs = {k: v / total * 100 for k, v in probs.items()}
        return probs
