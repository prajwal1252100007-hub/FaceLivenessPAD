import numpy as np
from insightface.app import FaceAnalysis


class FaceMatcher:

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

    def get_embedding(self, frame):
        faces = self.app.get(frame)

        if not faces:
            return None

        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        embedding = face.embedding.astype(
            np.float32
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            return None

        return embedding / norm

    def compare(
        self,
        embedding1,
        embedding2
    ):
        embedding1 = np.asarray(
            embedding1,
            dtype=np.float32
        )

        embedding2 = np.asarray(
            embedding2,
            dtype=np.float32
        )

        norm1 = np.linalg.norm(
            embedding1
        )

        norm2 = np.linalg.norm(
            embedding2
        )

        if norm1 == 0 or norm2 == 0:
            return 0.0

        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2

        return float(
            np.dot(
                embedding1,
                embedding2
            )
        )