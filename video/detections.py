import logging
import sys
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Tuple, Iterator, Optional, Dict

import numpy as np
import face_recognition
import cv2
import torch
from sklearn.cluster import DBSCAN

from .db import PersonDB, PersonItem

logger = logging.getLogger(__name__)


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
    resize_rate=1.,
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


@dataclass
class Person:
    timestamps: List[float]
    ratio: float
    db_item: Optional[PersonItem]


@dataclass
class MetaData:
    persons: List[Person]
    objects: Dict[float, List[str]]


def detect_faces(frame):
    locations = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, locations)
    return list(zip(locations, encodings))


def detect_objects(yolo_model, frame, min_confidence: float) -> List[str]:
    detection = yolo_model([frame])
    objs = detection.xywh[0].cpu().data.numpy()
    confidence, class_idx = objs[:, 4], objs[:, 5]
    mask = (np.isin(class_idx, class_idx)) & np.array(confidence >= min_confidence)
    return [
        yolo_model.names[int(class_idx)]
        for class_idx in class_idx[mask]
    ]


def process_file(
    filepath: str,
    db_path: str,
    sampling_rate: int = 4,
    resize_rate: float = 1,
    max_len=sys.maxsize,
) -> MetaData:
    db = PersonDB(db_path)
    yolo_model = torch.hub.load('ultralytics/yolov5', model='yolov5s', pretrained=True)

    faces_timeline = FacesTimeline()
    total_frames = 0
    objects_timestamps = defaultdict(list)

    for timestamp, frame in iter_frames(filepath, sampling_rate=sampling_rate, resize_rate=resize_rate):
        if timestamp > max_len:
            break

        for face_location, face_encoding in detect_faces(frame):
            faces_timeline.timestamps.append(timestamp)
            faces_timeline.locations.append(face_location)
            faces_timeline.encodings.append(face_encoding)

        for object_name in detect_objects(yolo_model, frame, min_confidence=0.65):
            objects_timestamps[timestamp].append(object_name)

        total_frames += 1
        logger.info('Processed frame %s', total_frames)

    labels = clusterize_encodings(faces_timeline.encodings)

    mean_encodings = {
        label: np.stack(faces_timeline.encodings)[labels == label].mean(axis=0)
        for label in np.unique(labels)
    }

    known_persons = {}
    for label, face_encoding in mean_encodings.items():
        person = db.find(face_encoding)
        if person is not None:
            known_persons[label] = person

    person_timestamps = defaultdict(list)
    for timestamp, label in zip(faces_timeline.timestamps, labels):
        person_timestamps[label].append(timestamp)

    return MetaData(
        persons=[
            Person(
                timestamps=timestamps,
                ratio=len(timestamps) / total_frames,
                db_item=known_persons.get(label),
            )
            for label, timestamps in person_timestamps.items()
        ],
        objects=objects_timestamps,
    )
