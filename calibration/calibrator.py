from calibration import origin_detection
from chessboard_corner_detection import ChessboardCornerDetector
from image_correction import ImageCorrector
from origin_detection import OriginDetector
class Calibrator:
    def __init__(self):
        ccDetector = ChessboardCornerDetector()
        imCorrector = ImageCorrector()
        orDetector = OriginDetector()
    def fit(self, args):
        pass
    def predict(self, args):
        pass
    def test(self, args):
        pass
