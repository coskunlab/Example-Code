from skimage import exposure, util
from skimage.feature import blob_log
from scipy.spatial import cKDTree
from skimage.morphology import white_tophat as skimage_white_tophat
from skimage.exposure import rescale_intensity
import skimage.util
import h5py
import numpy as np
import pickle
import os 
from tqdm.auto import tqdm

class PLA_detection(object):
    def __init__(self, path):
        self.path = path
        self.get_markers()
        self.PPI_dict = {}

    def detect_spot(self, marker, thres=.05, min_radius=1, max_radius=4):
        # Get PLA channel image
        print(f'Reading image {marker}')
        img = self.get_image_from_marker(marker)
        img = img[np.newaxis, ...]

        print(f'Processing image {marker}')
        # White top hat filtering
        img_wth = self.white_tophat(img, 3)
        img_wth = skimage.util.img_as_ubyte(img_wth)

        # Get spots counts
        spots_filtered = self.get_spots(img_wth, thres=thres, min_radius=min_radius, max_radius=max_radius)

        img_spot = self.plot_spot_on_image(img_wth, spots_filtered, 1, 2) 
        print(np.unique(img_spot[0], return_counts=True))
        self.PPI_dict[marker] = spots_filtered

        return img_spot, img_wth, spots_filtered, img

    def save_pickle(self, path):
        try:
            os.remove(path)
            print("File exist. Deleted")
        except FileNotFoundError:
            pass

        # Open a file and use dump()
        with open(path, 'wb') as file:
            pickle.dump(self.PPI_dict, file, pickle.HIGHEST_PROTOCOL)

    def get_image_from_marker(self, marker):
        indice = self.get_indice(marker)
        with h5py.File(self.path, "r") as f:
            img = f['imgs'][indice]
        return util.img_as_ubyte(img)
    
    def get_markers(self):
        with h5py.File(self.path, "r") as f:
            markers = f['imgs'].attrs['Marker']
        self.markers = markers

    def get_indice(self, marker):
        indice = list(self.markers).index(marker)
        return indice

    
    def get_spots(self, imgs, min_radius=1, max_radius=3, num_sigma=5, thres=.3):
        spots_all = [] 
        for img in imgs:
            spots = self.detect_spots_log(img[np.newaxis,...], min_radius, max_radius, num_sigma=num_sigma, threshold=thres)
            spots_all.append(spots)

        for i in range(len(spots_all)):
            spots_all[i][:,0,...] = i

        spots = np.concatenate(spots_all, axis=0)
        return spots
    
    @staticmethod
    def contrast_str(img, n_min=0, n_max=99.9):
        p2, p98 = np.percentile(img, (n_min, n_max))
        img_rescale = rescale_intensity(img, in_range=(p2, p98))
        img_rescale = util.img_as_ubyte(img_rescale)
        return img_rescale
    
    @staticmethod
    def white_tophat(image, radius):
        # ensure iterable radius
        if not isinstance(radius, (tuple, list, np.ndarray)):
            radius = (radius,)*image.ndim

        # convert to footprint shape
        shape = [2*r+1 for r in radius]

        # run white tophat
        return skimage_white_tophat(image, footprint=np.ones(shape))

    @staticmethod
    def detect_spots_log(image, min_radius, max_radius, num_sigma=5, **kwargs):
        # ensure iterable radii
        if not isinstance(min_radius, (tuple, list, np.ndarray)):
            min_radius = (min_radius,)*image.ndim
        if not isinstance(max_radius, (tuple, list, np.ndarray)):
            max_radius = (max_radius,)*image.ndim

        # compute defaults
        min_radius = np.array(min_radius)
        max_radius = np.array(max_radius)
        min_sigma = 0.8 * min_radius / np.sqrt(image.ndim)
        max_sigma = 1.2 * max_radius / np.sqrt(image.ndim)

        # set given arguments
        kwargs['min_sigma'] = min_sigma
        kwargs['max_sigma'] = max_sigma
        kwargs['num_sigma'] = num_sigma

        # set additional defaults
        if 'overlap' not in kwargs:
            kwargs['overlap'] = 1.0
        if 'threshold' not in kwargs:
            kwargs['threshold'] = None
            kwargs['threshold_rel'] = 0.1
        # run
        return blob_log(image, **kwargs)

    @staticmethod
    def plot_spot_on_image(reference, spots, spacing, radius):
        spot_img = np.zeros_like(reference)
        coords = (spots[:, :3] / spacing).astype(int)
        r = radius  # shorthand
        for coord in coords:
            slc = tuple(slice(x-r, x+r) for x in coord)
            spot_img[slc] = 1
        return spot_img

    @staticmethod
    def apply_foreground_mask(spots, mask, ratio):
        """
        """

        # get spot locations in mask voxel coordinates
        x = np.round(spots[:, :3] * ratio).astype(np.uint16)

        # correct out of range rounding errors
        for i in range(3):
            x[x[:, i] >= mask.shape[i], i] = mask.shape[i] - 1

        # filter spots and return
        return spots[mask[x[:, 0], x[:, 1], x[:, 2]] > 1]

class PLA_detection_3D(object):
    def __init__(self, paths):
        self.paths = paths
        self.get_markers()
        self.PPI_dict = {}

    def detect_spot(self, marker, thres=.05, min_radius=1, max_radius=4):
        # Get PLA channel image
        imgs = self.get_image_from_marker(marker)

        # White top hat filtering
        imgs_wth = [skimage.util.img_as_ubyte(self.white_tophat(img, 5)) for img in imgs]
        imgs_wth = np.stack(imgs_wth)

        # Get spots counts
        spots_filtered = self.get_spots(imgs_wth, thres=thres, min_radius=min_radius, max_radius=max_radius)

        img_spot = self.plot_spot_on_image(imgs_wth, spots_filtered, 1, 2) 
        print(np.unique(img_spot[0], return_counts=True))
        self.PPI_dict[marker] = spots_filtered

        return img_spot, imgs_wth, spots_filtered, imgs

    def save_pickle(self, path):
        try:
            os.remove(path)
            print("File exist. Deleted")
        except FileNotFoundError:
            pass

        # Open a file and use dump()
        with open(path, 'wb') as file:
            pickle.dump(self.PPI_dict, file, pickle.HIGHEST_PROTOCOL)

    def get_image_from_marker(self, marker):
        indice = self.get_indice(marker)
        imgs = []
        for path in tqdm(self.paths, desc='Reading images', leave=False):
            with h5py.File(path, "r") as f:
                img = f['imgs'][indice]
            imgs.append(img)
        imgs = self.make_imgs_same_dim(imgs)
        return imgs

    def get_markers(self):
        with h5py.File(self.paths[0], "r") as f:
            markers = f['imgs'].attrs['Marker']
        self.markers = markers

    def get_indice(self, marker):
        indice = list(self.markers).index(marker)
        return indice
 
    def get_spots(self, imgs, min_radius=1, max_radius=3, num_sigma=5, thres=.3):
        spots_all = [] 
        for img in tqdm(imgs, total=len(imgs), desc='Detecting PPI spots', leave=False):
            spots = self.detect_spots_log(img[np.newaxis,...], min_radius, max_radius, num_sigma=num_sigma, threshold=thres)
            spots_all.append(spots)

        for i in range(len(spots_all)):
            spots_all[i][:,0,...] = i

        spots = np.concatenate(spots_all, axis=0)
        return spots

    @staticmethod
    def make_imgs_same_dim(imgs):
        # Get max dimensions
        shapes = np.array([img.shape for img in imgs])
        min_x, min_y = shapes.min(axis=0)
        return [img[:min_x, :min_y] for img in imgs]

    @staticmethod
    def white_tophat(image, radius):
        # ensure iterable radius
        if not isinstance(radius, (tuple, list, np.ndarray)):
            radius = (radius,)*image.ndim

        # convert to footprint shape
        shape = [2*r+1 for r in radius]

        # run white tophat
        return skimage_white_tophat(image, footprint=np.ones(shape))

    @staticmethod
    def detect_spots_log(image, min_radius, max_radius, num_sigma=5, **kwargs):
        # ensure iterable radii
        if not isinstance(min_radius, (tuple, list, np.ndarray)):
            min_radius = (min_radius,)*image.ndim
        if not isinstance(max_radius, (tuple, list, np.ndarray)):
            max_radius = (max_radius,)*image.ndim

        # compute defaults
        min_radius = np.array(min_radius)
        max_radius = np.array(max_radius)
        min_sigma = 0.8 * min_radius / np.sqrt(image.ndim)
        max_sigma = 1.2 * max_radius / np.sqrt(image.ndim)

        # set given arguments
        kwargs['min_sigma'] = min_sigma
        kwargs['max_sigma'] = max_sigma
        kwargs['num_sigma'] = num_sigma

        # set additional defaults
        if 'overlap' not in kwargs:
            kwargs['overlap'] = 1.0
        if 'threshold' not in kwargs:
            kwargs['threshold'] = None
            kwargs['threshold_rel'] = 0.1
        # run
        return blob_log(image, **kwargs)

    @staticmethod
    def plot_spot_on_image(reference, spots, spacing, radius):
        spot_img = np.zeros_like(reference)
        coords = (spots[:, :3] / spacing).astype(int)
        r = radius  # shorthand
        for coord in coords:
            slc = tuple(slice(x-r, x+r) for x in coord)
            spot_img[slc] = 1
        return spot_img

    @staticmethod
    def apply_foreground_mask(spots, mask, ratio):
        """
        """

        # get spot locations in mask voxel coordinates
        x = np.round(spots[:, :3] * ratio).astype(np.uint16)

        # correct out of range rounding errors
        for i in range(3):
            x[x[:, i] >= mask.shape[i], i] = mask.shape[i] - 1

        # filter spots and return
        return spots[mask[x[:, 0], x[:, 1], x[:, 2]] > 1]

# class PLA_detection_3D(object):
#     def __init__(self, paths):
#         self.paths = paths
#         self.get_markers()
#         self.PPI_dict = {}

#     def save_pickle(self, path):
#         try:
#             os.remove(path)
#             print("File exist. Deleted")
#         except FileNotFoundError:
#             pass

#         # Open a file and use dump()
#         with open(path, 'wb') as file:
#             pickle.dump(self.PPI_dict, file, pickle.HIGHEST_PROTOCOL)

#     def get_image_from_marker(self, marker):
#         print('Reading images')
#         indice = self.get_indice(marker)
#         imgs = []
#         for path in self.paths:
#             with h5py.File(path, "r") as f:
#                 img = f['imgs'][indice]
#             imgs.append(self.contrast_str(img))
#         imgs = self.make_imgs_same_dim(imgs)
#         return np.stack(imgs)

#     def get_markers(self):
#         with h5py.File(self.paths[0], "r") as f:
#             markers = f['imgs'].attrs['Marker']
#         self.markers = markers

#     def get_indice(self, marker):
#         indice = list(self.markers).index(marker)
#         return indice

#     def detect_spot(self, marker, thres=.05, min_radius=1, max_radius=4):
#         img_spots = []
#         img_wths = []
#         spots = []
#         for path in self.paths:
#             pla_detect = PLA_detection(path)
#             img_spot, img_wth, spots_filtered, _ = pla_detect.detect_spot(marker, thres=thres, min_radius=min_radius, max_radius=max_radius)
#             img_spots.append(img_spot)
#             img_wths.append(img_wth)
#             spots.append(spots_filtered)
#         return self.make_imgs_same_dim(img_spots), self.make_imgs_same_dim(img_wths), spots

#     @staticmethod
#     def make_imgs_same_dim(imgs):
#         # Get max dimensions
#         shapes = np.array([img.shape for img in imgs])
#         min_x, min_y = shapes.min(axis=0)
#         return [img[:min_x, :min_y] for img in imgs]
