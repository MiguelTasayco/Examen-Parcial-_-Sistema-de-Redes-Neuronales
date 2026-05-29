"""
compliance_checker.py
Módulo de posprocesamiento para verificación de cumplimiento de uso de casco.
Recibe detecciones por fotograma y devuelve estado de cumplimiento y tasa global.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2


# ── Configuración de reglas (ajustable sin reentrenar) ───────────────────────
COMPLIANCE_RULES = {
    "min_confidence":   0.40,   # Umbral mínimo de confianza para considerar una detección
    "requires_helmet":  True,   # Clase 0 (helmet) es requerida
    "flag_no_helmet":   True,   # Clase 1 (no_helmet) activa no-conformidad directa
    "flag_person":      False,  # Clase 2 (person sin info EPP): no penalizar por defecto
    "non_compliant_color": (220, 50, 50),    # BGR rojo para bboxes no conformes
    "compliant_color":     (50, 180, 100),   # BGR verde para bboxes conformes
    "uncertain_color":     (220, 160, 50),   # BGR amarillo para clase 'person'
}


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2 en píxeles
    is_compliant: Optional[bool] = None


@dataclass
class FrameResult:
    frame_id: int
    detections: List[Detection]
    n_helmet:    int = 0
    n_no_helmet: int = 0
    n_person:    int = 0
    n_compliant:     int = 0
    n_non_compliant: int = 0
    compliance_rate: float = 1.0  # 1.0 = 100% conforme


CLASS_NAMES = {0: 'helmet', 1: 'no_helmet', 2: 'person'}


def evaluate_detection(det: Detection, rules: dict) -> Detection:
    """Aplica las reglas de cumplimiento a una detección individual."""
    if det.confidence < rules["min_confidence"]:
        det.is_compliant = None  # Ignorar detecciones de baja confianza
        return det

    if det.cls_id == 0:  # helmet
        det.is_compliant = True
    elif det.cls_id == 1:  # no_helmet
        det.is_compliant = False
    elif det.cls_id == 2:  # person
        det.is_compliant = None if not rules["flag_person"] else False

    return det


def process_frame(frame_id: int, raw_detections: List[dict],
                  rules: dict = COMPLIANCE_RULES) -> FrameResult:
    """
    Procesa las detecciones de un fotograma y calcula el estado de cumplimiento.

    Args:
        frame_id: Número del fotograma.
        raw_detections: Lista de dicts con keys: cls_id, confidence, bbox_xyxy.
        rules: Diccionario de reglas de cumplimiento.

    Returns:
        FrameResult con métricas de cumplimiento del fotograma.
    """
    detections = []
    for d in raw_detections:
        det = Detection(
            cls_id=d['cls_id'],
            cls_name=CLASS_NAMES.get(d['cls_id'], 'unknown'),
            confidence=d['confidence'],
            bbox_xyxy=tuple(d['bbox_xyxy']),
        )
        det = evaluate_detection(det, rules)
        detections.append(det)

    result = FrameResult(frame_id=frame_id, detections=detections)

    for det in detections:
        if det.cls_id == 0: result.n_helmet    += 1
        elif det.cls_id == 1: result.n_no_helmet += 1
        elif det.cls_id == 2: result.n_person    += 1

        if det.is_compliant is True:
            result.n_compliant += 1
        elif det.is_compliant is False:
            result.n_non_compliant += 1

    total = result.n_compliant + result.n_non_compliant
    result.compliance_rate = result.n_compliant / total if total > 0 else 1.0

    return result


def annotate_frame(frame: np.ndarray, result: FrameResult,
                   rules: dict = COMPLIANCE_RULES) -> np.ndarray:
    """
    Dibuja bounding boxes coloreados según estado de cumplimiento en el fotograma.
    Verde = conforme, Rojo = no conforme, Amarillo = incierto.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for det in result.detections:
        if det.confidence < rules["min_confidence"]:
            continue

        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]

        if det.is_compliant is True:
            color = rules["compliant_color"]
        elif det.is_compliant is False:
            color = rules["non_compliant_color"]
        else:
            color = rules["uncertain_color"]

        # Bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Etiqueta
        label = f"{det.cls_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Overlay de tasa de cumplimiento
    rate_pct = result.compliance_rate * 100
    rate_color = (50, 180, 100) if rate_pct >= 80 else \
                 (220, 160, 50) if rate_pct >= 50 else (220, 50, 50)
    overlay_text = f"Cumplimiento: {rate_pct:.0f}%  |  Sin casco: {result.n_no_helmet}"
    cv2.rectangle(annotated, (0, 0), (w, 30), (20, 20, 20), -1)
    cv2.putText(annotated, overlay_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rate_color, 2, cv2.LINE_AA)

    return annotated


def summarize_session(frame_results: List[FrameResult]) -> dict:
    """Genera resumen de una sesión completa (múltiples fotogramas)."""
    if not frame_results:
        return {}
    rates = [r.compliance_rate for r in frame_results]
    total_no_helmet = sum(r.n_no_helmet for r in frame_results)
    return {
        'total_frames':       len(frame_results),
        'mean_compliance':    round(np.mean(rates), 4),
        'min_compliance':     round(np.min(rates), 4),
        'frames_non_compliant': sum(1 for r in frame_results if r.n_no_helmet > 0),
        'total_no_helmet_detections': total_no_helmet,
    }

print('compliance_checker.py escrito correctamente.')
