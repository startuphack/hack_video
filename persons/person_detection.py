from dataclasses import dataclass, field
from typing import List, Tuple, Iterator

import numpy as np
import face_recognition
import cv2
from sklearn.cluster import DBSCAN

from .db import PersonDB


@dataclass
class FacesTimeline:
    """
    timestamps: Метка времени в секундах
    locations: Координаты лиц в формате (top, right, bottom, left)
    encodings: float64 вектора размерностью (128,)
    """
    timestamps: List[float] = field(default_factory=list)
    locations: List[Tuple[int, int, int, int]] = field(default_factory=list)
    encodings: List[np.ndarray] = field(default_factory=list)


def iter_frames(
    filepath: str, 
    sampling_rate: int = 10,
    resize_rate = 1.,
) -> Iterator[Tuple[float, np.ndarray]]:
    assert 0 < resize_rate <= 1

    cap = cv2.VideoCapture(filepath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_number = 0
    try:
        while True:
            rv, frame = cap.read()
            if not rv:
                break

            timestamp = round(frame_number / fps, 3)
            frame_number += 1
            if frame_number % sampling_rate != 0:
                continue

            if resize_rate < 1:
                frame = cv2.resize(frame, (0, 0), fx=resize_rate, fy=resize_rate)

            rgb_frame = frame[:, :, ::-1]  # BGR -> RGB
            yield timestamp, rgb_frame

    finally:
        cap.release()


def clusterize_encodings(encodings: List[np.ndarray], **kwargs) -> np.ndarray:
    model = DBSCAN(**kwargs)
    return model.fit_predict(encodings)


def process_file(filepath: str):
    db = PersonDB('persons.db')
    timeline = FacesTimeline()

    for timestamp, frame in iter_frames(filepath, sampling_rate=100):

        _locations = face_recognition.face_locations(frame)
        _encodings = face_recognition.face_encodings(frame, _locations)
        for location, encoding in zip(_locations, _encodings):
            timeline.timestamps.append(timestamp)
            timeline.locations.append(location)
            timeline.encodings.append(encoding)

    labels = clusterize_encodings(timeline.encodings)
    unique_labels, counts = np.unique(labels, return_counts=True)

    mean_encodings = [
        np.stack(timeline.encodings)[labels == label].mean(axis=0)
        for label in unique_labels
    ]
    for idx, encoding in enumerate(mean_encodings):
        person = db.find(encoding)
        if person is None:
            continue

    # todo
