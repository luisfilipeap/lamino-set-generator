from inline_setup_3D import *
import astra
import time
from imageio import imread
from matplotlib import pyplot as plt
from scipy import misc


class ScanningObject:
    """
    This class defines an inline scanning geometry and executes image reconstructions.
    Attributes
    ----------
    proj_geom   : dict
        It holds the projection geometry to be used;
    setup       : ndarray
        It holds the scanning geometry to be used;
    vol_geom    : dict
        It holds the characteristics of the reconstruction volume;
    Methods
    -------
    run(phantom_param, rec_algorithm_param='SIRT_CUDA', n_iterations_param=100)
        It executes an image reconstruction using the projections acquired in the inline setup.
    """

    def __init__(self, alpha_param, n_cells_param, n_proj_param, rec_size_param=256):
        """
        It creates a new instance of the class ScanningExecution.
        :param alpha_param: fan-beam opening angle of the X-ray source used in the inline CT setup;
        :param n_cells_param: number of detector elements used in the inline CT setup;
        :param n_proj_param: number of X-ray projections aquired during the object movement;
        :param rec_size_param: number W of pixels of the W x W reconstruction grid;

        acquisition.
        """

        self.vol_geom = astra.create_vol_geom(128, 128, 10)

        self.setup = InlineScanningSetup(alpha=alpha_param, detector_cells=n_cells_param,
                                         number_of_projections=n_proj_param, object_size=rec_size_param)

        self.proj_geom = astra.create_proj_geom('cone_vec', n_cells_param, n_cells_param, self.setup.get_geometry_matrix())

    def run(self, phantom_param, rec_algorithm_param='SIRT3D_CUDA', n_iterations_param=100):
        """
        It executes an image reconstruction using the projections acquired in the inline setup.
        :param phantom_param: 3D volume of the phantom that should be used to simulate the acquisition of projections from
        real object;
        :param rec_algorithm_param: reconstruction algorithm to be used. The option available are: SIRT_CUDA and FBP_CUDA;
        :param n_iterations_param: number of iterations to be used in case of iterative reconstructions;
        :return: a dictionary containing the reconstructed image into 'rec' index, the reconstruction time into 'time' index,
        and the acquired sinogram into the 'sino' index;
        """


        #proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        proj_id, proj_data = astra.create_sino3d_gpu (phantom_param, self.proj_geom, self.vol_geom)

        #plt.figure("sino")
        #plt.imshow(proj_data[:,5,:])
        #plt.show()

        rec_id = astra.data3d.create('-vol', self.vol_geom)
        cfg = astra.astra_dict(rec_algorithm_param)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg)

        if rec_algorithm_param == 'SIRT3D_CUDA':
            start_time = time.time()
            astra.algorithm.run(alg_id, n_iterations_param)
            elapsed_time = time.time() - start_time
        else:
            astra.algorithm.run(alg_id)
            start_time = time.time()
            elapsed_time = time.time() - start_time

        output = {'rec': astra.data3d.get(rec_id), 'time': elapsed_time, 'sino': proj_data}



        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)


        return output


if __name__ == '__main__':

    #test code by running scanning_object.py
    src = "D:\\Datasets\\demo_data_plates\\plate_00000\\"

    plane = np.zeros((30, 300, 300))
    for k in range(30):
        i = imread(src + 'slice_{}.png'.format(k), pilmode='F')
        plane[k,:,:] = i

    setup = ScanningObject(alpha_param=30, n_cells_param=1024, n_proj_param=8, rec_size_param=256)
    out = setup.run(plane)

    plt.figure("REC")
    plt.imshow(out['sino'][:, 4,:])

    #plt.figure("PHANTOM")
    #plt.imshow(plane[0,:,:])

    plt.show()