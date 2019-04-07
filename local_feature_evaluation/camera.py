# Simple COLMAP camera class.
class Camera:
    def __init__(self):
        self.camera_model = None
        self.intrinsics = None
        self.qvec = None
        self.t = None

    def set_intrinsics(self, camera_model, intrinsics):
        self.camera_model = camera_model
        self.intrinsics = intrinsics

    def set_pose(self, qvec, t):
        self.qvec = qvec
        self.t = t
