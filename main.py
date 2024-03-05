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

    def __init__(self,cameraIntrinsicMatrix,listOfTransformationMatrices):
        self.setCameraIntrinsic(cameraIntrinsicMatrix)
        self.setlistOfTransformationMatrices(listOfTransformationMatrices)
        self.__rainbow_r = cm.get_cmap('rainbow_r', lut=100)
        self.__get_color = lambda z: [val for val in self.__rainbow_r(int(z.round()))[:3]]

    def applyLidarToCameraProjections(self,inputDirectory,outputDirectory,imageshape=(376, 1241,3),remove_plane = True,remove_outliers=True,color_map = "",cameraDirection = CameraDirection.POSITIVE):
        if color_map == "":
            color_map = self.__get_color
        image = np.zeros(imageshape)

        operand = self.__computeTotalMatrix()
        paths = sorted(glob(os.path.join(inputDirectory, '*.bin')))

        for index in range(len(paths)):
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

            # remove outliers (points outside of the image frame)
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
                cv2.circle(image, (int(u[i]), int(v[i])), 1,color_map(z[i]), -1)
            matplotlib.image.imsave("{}/{}.png".format(outputDirectory,inputPath.split("/")[-1]),image)


if __name__ == '__main__':
    projectionTool = ProjectionTool(np.array([\
    [607.48,     -718.54,     -10.188,     -95.573],\
    [180.03,      5.8992,     -720.15,     -93.457],\
    [0.99997,  0.00048595,  -0.0072069,    -0.28464]]\
    ),[np.identity(4)])
    projectionTool.applyLidarToCameraProjections("./input","./output",)
