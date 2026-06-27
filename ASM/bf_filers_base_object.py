import numpy as np

class beamFormerBasedFilters:
    def __init__(self, grid, fs=1600, filter_len=0.032):
        self.grid_az, self.grid_al = grid['azimuth'], grid['elevation']
        self.fs = fs
        self.filter_len = filter_len
        self.N_filter = int(self.fs * self.filter_len)
        self.mic_array = None
        self.hrtf = None
        self.sm = None

    def set_hrtf(self, use_default_hrtf=True):
        if use_default_hrtf:
            from spatial_audio_objects.default_hrtf_object import defaultHrftObject
            hrtf = defaultHrftObject()
            hrtf.change_sampling_scheme(grid_out={'azimuth': self.grid_az, 'elevation': self.grid_al})
            hrtf.resample(fs_out=self.fs)
            hrtf.time_zero_pad(self.N_filter)
            hrtf.toFreq()
            self.hrtf = hrtf
        else:
            raise NotImplementedError
        return

    def set_array(self, steering_matrix_object):
        self.sm = steering_matrix_object
        self.sm.change_sampling_scheme(grid_out={'azimuth': self.grid_az, 'elevation': self.grid_al})
        return

    def plot(self, data):
        import matplotlib.pyplot as plt
        plt.plot(data)
        plt.show()

    @staticmethod
    def _shmat(N: int, azimuth: np.ndarray, elevation: np.ndarray):
        from utils.acl_utils import compute_spherical_harmonics_matrix as shmat
        return shmat(N, phi=azimuth, theta=elevation)