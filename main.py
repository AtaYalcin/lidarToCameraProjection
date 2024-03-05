import matplotlib.image
import numpy as np
from enum import Enum
from glob import glob
import os
import cv2
from sklearn import linear_model
from matplotlib import cm
from matplotlib import image

class ProjectionTool:
    class CameraDirection(Enum):
        POSITIVE = 0
        NEGATIVE = 1
    class ColorFunctionGenerator():
         def rainbow(lut):
             rainbow_r = cm.get_cmap('rainbow_r', lut=lut)
             return lambda z: [val for val in rainbow_r(int(z.round()))[:3]]

    def __computeTotalMatrix(self):
        result = np.identity(4)
        for matrix in self.__listOfTransformationMatrices:
            result = np.matmul(matrix,result)
        result = np.matmul(self.__cameraIntrinsicMatrix,result)
        return result
    def setCameraIntrinsic(self,cameraIntrinsicMatrix):
        self.__cameraIntrinsicMatrix = cameraIntrinsicMatrix
    def setlistOfTransformationMatrices(self,listOfTransformationMatrices):
        self.__listOfTransformationMatrices = listOfTransformationMatrices
    def setColorFunction(self,color_function):
        self.__color_function = color_function

    def __init__(self,cameraIntrinsicMatrix,listOfTransformationMatrices,color_function=""):
        self.setCameraIntrinsic(cameraIntrinsicMatrix)
        self.setlistOfTransformationMatrices(listOfTransformationMatrices)
        if color_function == "":
            rainbow_r = cm.get_cmap('rainbow_r', lut=100)
            self.__color_function = lambda z: [val for val in rainbow_r(int(z.round()))[:3]]
        else:
            self.__color_function = color_function

    def applyLidarToCameraProjections(self,inputDirectory,outputDirectory,imageshape=(376, 1241),remove_plane = True,remove_outliers=True,cameraDirection = CameraDirection.POSITIVE):
        operand = self.__computeTotalMatrix()
        paths = sorted(glob(os.path.join(inputDirectory, '*.bin')))

        for index in range(len(paths)):
            image = np.zeros((imageshape[0], imageshape[1], 3))
            inputPath = paths[index]
            scan_data = np.fromfile(inputPath, dtype=np.float32).reshape((-1, 4))
            xyz = scan_data[:, 0:3]

            xyz = np.delete(xyz, np.where(xyz[3, :] < 0), axis=1)

            if remove_plane:
                ransac = linear_model.RANSACRegressor(
                    linear_model.LinearRegression(),
                    residual_threshold=0.1,
                    max_trials=5000
                )

                X = xyz[:, :2]
                y = xyz[:, -1]
                ransac.fit(X, y)

                # remove outlier points (i.e. remove ground plane)
                mask = ransac.inlier_mask_
                xyz = xyz[~mask]

            xyzw = np.insert(xyz, 3, 1, axis=1).T

            camera = np.matmul(operand,xyzw)


            if cameraDirection == self.CameraDirection.POSITIVE:
                camera = np.delete(camera, np.where(camera[2, :] < 0)[0], axis=1)
            else:
                camera = np.delete(camera, np.where(camera[2, :] > 0)[0], axis=1)

            # get camera coordinates u,v,z
            camera[:2] /= camera[2, :]

            # remove outliers (points outside the image frame)
            if remove_outliers:
                u, v, z = camera
                img_h, img_w, _ = image.shape
                u_out = np.logical_or(u < 0, u > img_w)
                v_out = np.logical_or(v < 0, v > img_h)
                outlier = np.logical_or(u_out, v_out)
                camera = np.delete(camera, np.where(outlier), axis=1)

            uvz = camera

            u, v, z = uvz
            # draw LiDAR point cloud on blank image
            for i in range(len(u)):
                cv2.circle(image, (int(u[i]), int(v[i])), 1,self.__color_function(z[i]), -1)
            matplotlib.image.imsave("{}/{}.png".format(outputDirectory,inputPath.split("/")[-1]),image)


if __name__ == '__main__':
    #projectionTool = ProjectionTool(np.array([\
    #[607.48,     -718.54,     -10.188,     -95.573],\
    #[180.03,      5.8992,     -720.15,     -93.457],\
    #[0.99997,  0.00048595,  -0.0072069,    -0.28464]]\
    #),[np.identity(4)])


    T_velo_ref0 = np.array([[ 7.967514e-03, -9.999679e-01, -8.462264e-04, -1.377769e-02],\
       [-2.771053e-03,  8.241710e-04, -9.999958e-01, -5.542117e-02],\
       [ 9.999644e-01,  7.969825e-03, -2.764397e-03, -2.918589e-01],\
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
    T_ref0_ref2 = np.array([[ 9.999788e-01, -5.008404e-03, -4.151018e-03,  5.954406e-02],\
       [ 4.990516e-03,  9.999783e-01, -4.308488e-03, -7.675338e-04],\
       [ 4.172506e-03,  4.287682e-03,  9.999821e-01,  3.582565e-03],\
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
    R_ref0_rect2 = np.array([[ 9.999191e-01,  1.228161e-02, -3.316013e-03,  0.000000e+00],\
       [-1.228209e-02,  9.999246e-01, -1.245511e-04,  0.000000e+00],\
       [ 3.314233e-03,  1.652686e-04,  9.999945e-01,  0.000000e+00],\
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])

    intrinsics = np.array([[ 7.188560e+02,  0.000000e+00,  6.071928e+02,  4.538225e+01],\
       [ 0.000000e+00,  7.188560e+02,  1.852157e+02, -1.130887e-01],\
       [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  3.779761e-03]])
    projectionTool = ProjectionTool(intrinsics,[T_velo_ref0,T_ref0_ref2,R_ref0_rect2])

    projectionTool.setColorFunction(projectionTool.ColorFunctionGenerator.rainbow(50))
    projectionTool.applyLidarToCameraProjections("./input","./output")
