import numpy as np
from beam_former_based_filters.bf_filers_base_object import beamFormerBasedFilters
from beam_former_based_filters.analytical_error_base_class import analytical_error

class asm(beamFormerBasedFilters):
    def __init__(self, N, grid, fs=16000, filter_len=0.032):
        super().__init__(grid, fs, filter_len)
        self.N = N
        self._Cnm = None

    @property
    def Cnm(self):
        if self._Cnm is None:
            self._Cnm = self._calculate_coefficients()
        return self._Cnm

    def get_coefficients(self):
        return self.Cnm

    def _calculate_coefficients(self):
        if self.sm is None:
            raise ValueError("Steering matrix is not set.")

        V = self.sm.data
        Y = self._shmat(N=self.N, azimuth=self.grid_az, elevation=self.grid_al)
        self._Y = Y # store in cash for errors calculations.
        cnm = np.zeros(((self.N+1)**2, V.shape[1], V.shape[2]), dtype=np.complex64)
        cnm_ = np.zeros(((self.N + 1) ** 2, V.shape[1], V.shape[2]), dtype=np.complex64)
        from utils.optim.tikhonov import tikhonov
        for nm in range((self.N+1)**2):
            cnm_[nm,:] = np.array([tikhonov(A=V[:, f, :].conj(), b=Y[nm, :], lam=0.01) for f in range(V.shape[1])])
            cnm[nm,:] = np.array([np.linalg.lstsq(V[:, f, :].conj(), Y[nm, :], rcond=None)[0] for f in range(V.shape[1])])
        return cnm_

    def calc_ambisonics(self, mic_signals, Cnm=None, Cnm_domain='frequency'):
        if Cnm == None:
            Cnm = self.Cnm
        if Cnm_domain == 'frequency':
            # convert to time
            Cnm = np.fft.ifft(Cnm, axis=1)
        from utils.utils import convolve_and_sum
        rec_amb = convolve_and_sum(
            signal1=Cnm,
            signal2=mic_signals,
            time_dim1=1,
            time_dim2=1,
            channel_dims1=2,
            channel_dims2=0,
        )
        return rec_amb.T


class asmBinMse(analytical_error):
    def _assert_obj(self, asm_obj):
        if asm_obj is None:
            raise ValueError("ASM object is not set.")
        if not isinstance(asm_obj, asm):
            raise TypeError("asm_obj must be an instance of ASM.")
        if asm_obj.sm is None:
            raise ValueError("Steering matrix is not set in the ASM object.")
        return

    def requierd_attributes(self):
        self.obj.set_hrtf(use_default_hrtf=True)
        return ['c', 'hl', 'hnml','hr', 'hnmr', 'Y', 'V']

    def calculate_error(self, c, hl, hr, hnml, hnmr, Y, V):
        N = int(np.sqrt(c.shape[0]) - 1)
        hnml, hnmr = self._trim_hrtf_and_tilde(hnml, hnmr, N)

        msel = np.zeros(V.shape[1], np.float32)
        mser = np.zeros(V.shape[1], np.float32)
        for f in range(V.shape[1]):
            tmp = np.linalg.norm( hnml[:,f].T @ np.conj(c[:, f, :]) @ V[:, f, :].T - hl[:,f])
            msel[f] = np.square(tmp / np.linalg.norm(hl[:,f]))
            tmp = np.linalg.norm( hnmr[:,f].T @ np.conj(c[:, f, :]) @ V[:, f, :].T - hr[:,f])
            mser[f] = np.square(tmp / np.linalg.norm( hr[:,f]))
        return msel, mser

    def _acn(self, n, m):
        return n * n + n + m  # ACN

    def _tilde_flip(self, h, N):
        # h shape: ((N+1)**2, ...). Returns same shape.
        out = np.empty_like(h[:(N + 1) ** 2, ...])
        for n in range(N + 1):
            for m in range(-n, n + 1):
                out[self._acn(n, m)] = ((-1) ** m) * h[self._acn(n, -m)]
        return out

    def _trim_hrtf_and_tilde(self, hnml, hnmr, N):
        # Ensure first dim is SH channels in ACN order
        hnml = hnml[:(N + 1) ** 2, ...]
        hnmr = hnmr[:(N + 1) ** 2, ...]
        hnml = self._tilde_flip(hnml, N)
        hnmr = self._tilde_flip(hnmr, N)
        return hnml, hnmr

class asmMSE(analytical_error):

    def _assert_obj(self, asm_obj):
        if asm_obj is None:
            raise ValueError("ASM object is not set.")
        if not isinstance(asm_obj, asm):
            raise TypeError("asm_obj must be an instance of ASM.")
        if asm_obj.sm is None:
            raise ValueError("Steering matrix is not set in the ASM object.")
        return

    def requierd_attributes(self):
        return ['c', 'hl','hr', 'Y', 'V']

    def calculate_error(self, c, hl, hr, hnml, hnmr, Y, V):
        mse = np.zeros((c.shape[0], V.shape[1]))
        for nm in range(mse.shape[0]):
            for f in range(V.shape[1]):
                tmp = np.linalg.norm(np.conj(c[nm, f, :].T) @ V[:, f, :].T - np.conj(Y[nm, :]))
                mse[nm, f] = np.square(tmp / np.linalg.norm(Y[nm, :]))
                # mse[nm, f] = tmp
        return mse

class asmMagnitude(analytical_error):

    def _assert_obj(self, asm_obj):
        if asm_obj is None:
            raise ValueError("ASM object is not set.")
        if not isinstance(asm_obj, asm):
            raise TypeError("asm_obj must be an instance of ASM.")
        if asm_obj.sm is None:
            raise ValueError("Steering matrix is not set in the ASM object.")
        return

    def requierd_attributes(self):
        return ['c','Y', 'V']

    def calculate_error(self, c, hl, hr, hnml, hnmr, Y, V):
        amb_mag = np.zeros((c.shape[0], c.shape[1]))
        ref_mag = np.ones(c.shape[1]) * (Y[0,0])
        for nm in range(amb_mag.shape[0]):
            for f in range(amb_mag.shape[1]):
                amb_mag[nm, f] = np.linalg.norm(np.conj(c[nm, f, :].T) @ V[:, f, :].T)
        return ref_mag, amb_mag