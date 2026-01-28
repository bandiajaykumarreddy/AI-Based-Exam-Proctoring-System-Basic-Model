class ProctoringRules:
    def __init__(self):
        self.multi_face_frames = 0
        self.multi_face_confirmed = False
        self.multi_face_logged = False

        self.baseline_embeddings = []
        self.baseline = None

        self.impersonation_detected = False

    def check_multiple_faces(self, face_count, confirm_frames=5):
        if face_count > 1:
            self.multi_face_frames += 1
            if self.multi_face_frames >= confirm_frames:
                self.multi_face_confirmed = True
        else:
            self.multi_face_frames = 0

        return self.multi_face_confirmed

    def update_baseline(self, embedding, max_samples=20):
        if self.baseline is None:
            self.baseline_embeddings.append(embedding)
            if len(self.baseline_embeddings) >= max_samples:
                self.baseline = sum(self.baseline_embeddings) / len(self.baseline_embeddings)
        return self.baseline