import openface
import cv2

class FaceFinder:
    def __init__(self):
        self.align = openface.AlignDlib("models/dlib/shape_predictor_68_face_landmarks.dat")
        self.net = openface.TorchNeuralNet("models/openface/nn4.small2.v1.t7", 96)

    def getFaces(self, imgPath):
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        faces = []
        bbs = self.align.getAllFaceBoundingBoxes(rgbImg)

        for bb in bbs:
            face = {}
            
            if bb is None:
                raise Exception("Unable to find a face: {}".format(imgPath))
                
            alignedFace = self.align.align(96, rgbImg, bb,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                raise Exception("Unable to align image: {}".format(imgPath))

            rep = self.net.forward(alignedFace)
            
            face['rect'] = [bb.left(), bb.top(), bb.width(), bb.height()]
            face['rep'] = rep.tolist()
            face['size'] = bb.area()
            faces.append(face)
            
        return faces