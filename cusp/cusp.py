#!/usr/bin/env python3

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


cusp_spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('nbasis_functions', nb.int64),
    ('norm', nb.float64),
    ('s_mask', nb.float64[:]),
    ('shift', nb.float64[:, :]),
    ('orbital_sign', nb.int64[:, :]),
    ('r', nb.float64[:, :]),
    ('alpha', nb.float64[:, :, :]),
]


@nb.experimental.jitclass(cusp_spec)
class Cusp:

    def __init__(self, neu, ned, nbasis_functions):
        """
        Cusp
        """
        self.neu = neu
        self.ned = ned
        self.nbasis_functions = nbasis_functions
        # self.s_mask = s_mask
        # self.shift = shift
        # self.orbital_sign = orbital_sign
        # self.r = r
        # self.alpha = alpha
        self.norm = np.exp(np.math.lgamma(self.neu + 1) / self.neu / 2)
        self.s_mask = np.ones((self.nbasis_functions,))
        if self.neu == 1 and self.ned == 1:
            self.s_mask[:4] = 0.0
            # atoms, MO
            cusp_shift_up = cusp_shift_down = np.array([[0.0]])
            # atoms, MO - sign of s-type Gaussian functions centered on the nucleus
            cusp_orbital_sign_up = cusp_orbital_sign_down = np.array([[1]])
            # atoms, MO
            cusp_r_up = cusp_r_down = np.array([[0.4375]])
            # atoms, MO, alpha index
            cusp_alpha_up = cusp_alpha_down = np.array([[
                [0.29141713, -2.0, 0.25262478, -0.098352818, 0.11124336],
            ]])
        elif self.neu == 2 and self.ned == 2:
            self.s_mask[:5] = 0.0
            cusp_shift_up = cusp_shift_down = np.array([[0.0, 0.0]])
            cusp_orbital_sign_up = cusp_orbital_sign_down = np.array([[-1, -1]])
            cusp_r_up = cusp_r_down = np.array([[0.1205, 0.1180]])
            cusp_alpha_up = cusp_alpha_down = np.array([[
                [ 1.24736449, -4.0,  0.49675975, -0.30582868,  1.0897532],
                [-0.45510824, -4.0, -0.73882727, -0.89716308, -5.8491770]
            ]])
        elif self.neu == 5 and self.ned == 2:
            pass
        elif self.neu == 5 and self.ned == 5:
            self.s_mask[:5] = 0.0
            cusp_shift_up = cusp_shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            cusp_orbital_sign_up = cusp_orbital_sign_down = np.array([[1, 1, 0, 0, 0]])
            cusp_r_up = cusp_r_down = np.array([[0.0455, 0.0460, 0.0, 0.0, 0.0]])
            cusp_alpha_up = cusp_alpha_down = np.array([[
                [2.36314075, -10.0,  0.81732253,  0.15573932E+02, -0.15756663E+03],
                [0.91422900, -10.0, -0.84570201E+01, -0.26889022E+02, -0.17583628E+03],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
        elif self.neu == 9 and self.ned == 9:
            self.s_mask[:6] = 0.0
            cusp_shift_up = cusp_shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            cusp_orbital_sign_up = cusp_orbital_sign_down = np.array([[1, 1, 0, 0, 0, -1, 0, 0, 0]])
            cusp_r_up = cusp_r_down = np.array([[0.0205, 0.0200, 0, 0, 0, 0.0205, 0, 0, 0]])
            cusp_alpha_up = cusp_alpha_down = np.array([[
                [3.02622267, -18.0,  0.22734669E+01,  0.79076581E+02, -0.15595740E+04],
                [1.76719238, -18.0, -0.30835348E+02, -0.23112278E+03, -0.45351148E+03],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.60405204, -18, -0.35203155E+02, -0.13904842E+03, -0.35690426E+04],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
        elif self.neu == 18 and self.ned == 18:
            self.s_mask[:7] = 0.0
            cusp_shift_up = cusp_shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            cusp_orbital_sign_up = cusp_orbital_sign_down = np.array([[1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]])
            cusp_r_up = cusp_r_down = np.array([[0.0045, 0.0045, 0, 0, 0, 0.0045, 0, 0, 0, 0, 0, 0, 0, 0, 0.0045, 0, 0, 0]])
            cusp_alpha_up = cusp_alpha_down = np.array([[
                [3.77764947, -36.0,  0.22235586E+02, -0.56621947E+04, 0.62983424E+06],
                [2.62138667, -36.0, -0.12558804E+03, -0.72801257E+04, 0.58905979E+06],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.70814456, -36.0, -0.14280857E+03, -0.80481344E+04, 0.63438487E+06],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.56410983, -36.0, -0.14519895E+03, -0.85628812E+04, 0.69239963E+06],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
        elif self.neu == 12 and self.ned == 12:
            self.s_mask[:5] = 0.0
            self.s_mask[55:55+5] = 0.0
            self.s_mask[110:110+5] = 0.0
            cusp_shift_up = cusp_shift_down = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])
            cusp_orbital_sign_up = np.array([
                [-1, -1,  1, -1,  1, -1, 0, -1, -1, 0, -1,  1],
                [-1,  1, -1, -1, -1,  1, 0, -1, -1, 0,  1, -1],
                [-1, -1, -1, -1,  1,  1, 0, -1,  1, 0, -1, -1],
            ])
            cusp_orbital_sign_down = np.array([
                [-1, -1,  1,  1, -1, -1, 0, -1, -1, 0,  1, -1],
                [-1, -1, -1,  1, -1,  1, 0, -1,  1, 0,  1,  1],
                [-1,  1, -1,  1,  1,  1, 0, -1, -1, 0, -1,  1],
            ])
            cusp_r_up = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
            ])
            cusp_r_down = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
            ])
            cusp_alpha_up = np.array([
                [
                    [ 1.66696112, -0.80000242E+01,  0.72538040E+00,  0.74822749E+01, -0.59832829E+02],
                    [-3.67934712, -0.80068146E+01,  0.52306712E+00,  0.74024477E+01, -0.66331792E+02],
                    [-3.94191081, -0.79787241E+01,  0.77594866E+00,  0.17932088E+01, -0.10979109E+02],
                    [-0.14952920, -0.78746605E+01, -0.43071992E+01, -0.96038217E+01, -0.73806352E+02],
                    [-1.46038810, -0.79908365E+01, -0.50007568E+01, -0.10260692E+02, -0.95143069E+02],
                    [-0.36138067, -0.80924033E+01, -0.55877946E+01, -0.12746613E+02, -0.98421943E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.97822491, -0.82393021E+01, -0.65029776E+01, -0.21458170E+02, -0.62457982E+02],
                    [-4.22520946, -0.84474893E+01, -0.73574279E+01, -0.66355424E+01, -0.22665330E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-2.50170056, -0.83886665E+01, -0.73121006E+01, -0.21788297E+02, -0.94723331E+02],
                    [-1.55705345, -0.84492012E+01, -0.78325729E+01, -0.13722665E+02, -0.18661446E+03],
                ],
                [
                    [-3.69822013, -0.80095431E+01,  0.11302149E+01, -0.43409054E+01,  0.46442078E+02],
                    [ 1.66630435, -0.80000141E+01,  0.72320378E+00,  0.81261796E+01, -0.65047801E+02],
                    [-5.91498406, -0.80763347E+01,  0.90077681E+00,  0.31742107E+00,  0.35324292E+01],
                    [-0.46879152, -0.78900614E+01, -0.41907688E+01, -0.88090450E+01, -0.77431698E+02],
                    [-0.20295616, -0.79750959E+01, -0.44477740E+01, -0.11660536E+02, -0.61243691E+02],
                    [-0.60267378, -0.81319664E+01, -0.57277864E+01, -0.14396561E+02, -0.91691179E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.41424481, -0.83393407E+01, -0.69933925E+01, -0.15868367E+02, -0.13371167E+03],
                    [-1.69923582, -0.82051284E+01, -0.64601471E+01, -0.99697087E+01, -0.15524536E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-4.64757331, -0.62408167E+01,  0.11841009E+01, -0.30996021E+01,  0.22886716E+02],
                    [-3.64299934, -0.72053551E+01, -0.11465606E+01, -0.73818077E+01,  0.16054618E+00],
                ],
                [
                    [-3.97257763, -0.80089446E+01,  0.11382835E+01, -0.33320223E+01,  0.38514125E+02],
                    [-5.86803911, -0.79164537E+01,  0.92088515E+00, -0.96065381E+00,  0.16259254E+02],
                    [1.66760198,  -0.79999530E+01,  0.72544044E+00,  0.70833021E+01, -0.56621874E+02],
                    [-0.92086514, -0.78245771E+01, -0.39180783E+01, -0.73789685E+01, -0.76429773E+02],
                    [-0.35500196, -0.79059822E+01, -0.43591899E+01, -0.11018127E+02, -0.64511456E+02],
                    [-0.34258772, -0.80478675E+01, -0.55020049E+01, -0.69743091E+01, -0.14002539E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.61268484, -0.88953684E+01, -0.10239355E+02, -0.14393680E+02, -0.29851351E+03],
                    [-0.82873883, -0.82083731E+01, -0.64401279E+01, -0.99864145E+01, -0.15474953E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.42483045, -0.86471163E+01, -0.90649452E+01, -0.17907642E+02, -0.21389596E+03],
                    [-4.40492810, -0.52578816E+01,  0.37442246E+01,  0.60259540E+01,  0.24340638E+01],
                ],
            ])
            cusp_alpha_down = np.array([
                [
                    [ 1.66696112, -0.80000242E+01,  0.72538040E+00,  0.74822749E+01, -0.59832829E+02],
                    [-3.67934711, -0.80068146E+01,  0.52306718E+00,  0.74024468E+01, -0.66331787E+02],
                    [-3.94191082, -0.79787241E+01,  0.77594861E+00,  0.17932097E+01, -0.10979114E+02],
                    [-0.14952920, -0.78746605E+01, -0.43071992E+01, -0.96038217E+01, -0.73806352E+02],
                    [-1.46038810, -0.79908365E+01, -0.50007568E+01, -0.10260692E+02, -0.95143069E+02],
                    [-0.36138067, -0.80924033E+01, -0.55877946E+01, -0.12746613E+02, -0.98421943E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.97822491, -0.82393021E+01, -0.65029776E+01, -0.21458170E+02, -0.62457982E+02],
                    [-4.22520946, -0.84474893E+01, -0.73574279E+01, -0.66355420E+01, -0.22665330E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-2.50170056, -0.83886665E+01, -0.73121006E+01, -0.21788297E+02, -0.94723331E+02],
                    [-1.55705345, -0.84492012E+01, -0.78325729E+01, -0.13722665E+02, -0.18661446E+03],
                ],
                [
                    [-3.97257764, -0.80089446E+01,  0.11382835E+01, -0.33320231E+01,  0.38514130E+02],
                    [-5.86803910, -0.79164537E+01,  0.92088495E+00, -0.96065034E+00,  0.16259234E+02],
                    [ 1.66760198, -0.79999530E+01,  0.72544044E+00,  0.70833021E+01, -0.56621874E+02],
                    [-0.92086514, -0.78245771E+01, -0.39180783E+01, -0.73789685E+01, -0.76429773E+02],
                    [-0.35500196, -0.79059822E+01, -0.43591899E+01, -0.11018127E+02, -0.64511456E+02],
                    [-0.34258772, -0.80478675E+01, -0.55020049E+01, -0.69743091E+01, -0.14002539E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.61268483, -0.88953684E+01, -0.10239355E+02, -0.14393679E+02, -0.29851351E+03],
                    [-0.82873883, -0.82083731E+01, -0.64401279E+01, -0.99864145E+01, -0.15474953E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.42483045, -0.86471163E+01, -0.90649452E+01, -0.17907642E+02, -0.21389596E+03],
                    [-4.40492810, -0.52578816E+01,  0.37442246E+01,  0.60259539E+01,  0.24340639E+01],
                ],
                [
                    [-3.69822013, -0.80095431E+01,  0.11302149E+01, -0.43409047E+01,  0.46442075E+02],
                    [ 1.66630435, -0.80000141E+01,  0.72320378E+00,  0.81261796E+01, -0.65047801E+02],
                    [-5.91498405, -0.80763347E+01,  0.90077692E+00,  0.31741986E+00,  0.35324359E+01],
                    [-0.46879152, -0.78900614E+01, -0.41907688E+01, -0.88090450E+01, -0.77431698E+02],
                    [-0.20295616, -0.79750959E+01, -0.44477740E+01, -0.11660536E+02, -0.61243691E+02],
                    [-0.60267378, -0.81319664E+01, -0.57277864E+01, -0.14396561E+02, -0.91691179E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.41424481, -0.83393407E+01, -0.69933925E+01, -0.15868367E+02, -0.13371167E+03],
                    [-1.69923582, -0.82051284E+01, -0.64601471E+01, -0.99697088E+01, -0.15524535E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-4.64757331, -0.62408167E+01,  0.11841009E+01, -0.30996020E+01,  0.22886715E+02],
                    [-3.64299934, -0.72053551E+01, -0.11465606E+01, -0.73818076E+01,  0.16054597E+00],
                ]
            ])
        self.shift = np.concatenate((cusp_shift_up, cusp_shift_down), axis=1)
        self.orbital_sign = np.concatenate((cusp_orbital_sign_up, cusp_orbital_sign_down), axis=1)
        self.r = np.concatenate((cusp_r_up, cusp_r_down), axis=1)
        self.alpha = np.concatenate((cusp_alpha_up, cusp_alpha_down), axis=1)

    def wfn(self, n_vectors: np.ndarray):
        """Calculate cusped correction for s-part of orbitals.
        We apply a cusp correction to each orbital at each nucleus at which it is nonzero. Inside some cusp
        correction radius rc we replace φ, the part of the orbital arising from s-type Gaussian functions centred
        on the nucleus in question, by
        φ̃ = C + sign[φ̃(0)] * exp[p(r)]
        in gaussians.f90 set
        POLYPRINT=.true. ! Include cusp polynomial coefficients in CUSP_INFO output.
        """
        orbital = np.zeros((self.neu + self.ned, self.neu + self.ned))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.r[atom, i]:
                        orbital[i, j] = self.orbital_sign[atom, i] * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r**2 +
                            self.alpha[atom, i, 3] * r**3 +
                            self.alpha[atom, i, 4] * r**4
                        ) + self.shift[atom, i]

        for i in range(self.neu, self.neu + self.ned):
            for j in range(self.neu, self.neu + self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.r[atom, i]:
                        orbital[i, j] = self.orbital_sign[atom, i] * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r**2 +
                            self.alpha[atom, i, 3] * r**3 +
                            self.alpha[atom, i, 4] * r**4
                        ) + self.shift[atom, i]

        return self.norm * orbital

    def gradient(self, n_vectors: np.ndarray):
        """Cusp part of gradient"""
        gradient = np.zeros((self.neu + self.ned, self.neu + self.ned, 3))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.r[atom, i]:
                        gradient[i, j] = self.orbital_sign[atom, i] * (
                                self.alpha[atom, i, 1] +
                                2 * self.alpha[atom, i, 2] * r +
                                3 * self.alpha[atom, i, 3] * r**2 +
                                4 * self.alpha[atom, i, 4] * r**3
                        ) * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r ** 2 +
                            self.alpha[atom, i, 3] * r ** 3 +
                            self.alpha[atom, i, 4] * r ** 4
                        ) * n_vectors[atom, j] / r + self.shift[atom, i]

        for i in range(self.neu, self.neu + self.ned):
            for j in range(self.neu, self.neu + self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.r[atom, i]:
                        gradient[i, j] = self.orbital_sign[atom, i] * (
                                self.alpha[atom, i, 1] +
                                2 * self.alpha[atom, i, 2] * r +
                                3 * self.alpha[atom, i, 3] * r**2 +
                                4 * self.alpha[atom, i, 4] * r**3
                        ) * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r ** 2 +
                            self.alpha[atom, i, 3] * r ** 3 +
                            self.alpha[atom, i, 4] * r ** 4
                        ) * n_vectors[atom, j] / r + self.shift[atom, i]

        return self.norm * gradient

    def laplacian(self, n_vectors: np.ndarray):
        """Cusp part of laplacian"""
        laplacian = np.zeros((self.neu + self.ned, self.neu + self.ned))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.r[atom, i]:
                        laplacian[i, j] = self.orbital_sign[atom, i] * (
                            2 * self.alpha[atom, i, 1] +
                            4 * self.alpha[atom, i, 2] * r +
                            6 * self.alpha[atom, i, 3] * r**2 +
                            8 * self.alpha[atom, i, 4] * r**3 +
                            2 * r * (self.alpha[atom, i, 2] + 3 * self.alpha[atom, i, 3] * r + 6 * self.alpha[atom, i, 4] * r**2) +
                            r * (self.alpha[atom, i, 1] + 2 * self.alpha[atom, i, 2] * r + 3*self.alpha[atom, i, 3] * r**2 + 4*self.alpha[atom, i, 4] * r**3)**2
                        ) * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r ** 2 +
                            self.alpha[atom, i, 3] * r ** 3 +
                            self.alpha[atom, i, 4] * r ** 4
                        ) / r + self.shift[atom, i]

        for i in range(self.neu, self.neu + self.ned):
            for j in range(self.neu, self.neu + self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.r[atom, i]:
                        laplacian[i, j] = self.orbital_sign[atom, i] * (
                            2 * self.alpha[atom, i, 1] +
                            4 * self.alpha[atom, i, 2] * r +
                            6 * self.alpha[atom, i, 3] * r**2 +
                            8 * self.alpha[atom, i, 4] * r**3 +
                            2 * r * (self.alpha[atom, i, 2] + 3 * self.alpha[atom, i, 3] * r + 6 * self.alpha[atom, i, 4] * r**2) +
                            r * (self.alpha[atom, i, 1] + 2 * self.alpha[atom, i, 2] * r + 3*self.alpha[atom, i, 3] * r**2 + 4*self.alpha[atom, i, 4] * r**3)**2
                        ) * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r ** 2 +
                            self.alpha[atom, i, 3] * r ** 3 +
                            self.alpha[atom, i, 4] * r ** 4
                        ) / r + self.shift[atom, i]

        return self.norm * laplacian

    def hessian(self, n_vectors: np.ndarray):
        """Cusp part of laplacian"""
        hessian = np.zeros((self.neu + self.ned, self.neu + self.ned))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.r[atom, i]:
                        pass

        for i in range(self.ned):
            for j in range(self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, self.neu + j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.r[atom, i]:
                        pass

        return self.norm * hessian


def wfn_s(r, coefficients, exponents):
    """wfn of single electron of any s-orbital"""
    s_part = 0.0
    for primitive in range(coefficients.shape[0]):
        s_part += coefficients[primitive] * np.exp(-exponents[primitive] * r * r)
    return s_part


def fit(fit_function, xdata, ydata, p0, plot):
    """Fit gaussian basis by slater"""

    try:
        popt, pcov = curve_fit(fit_function, xdata, ydata, p0, maxfev=1000000)
        perr = np.sqrt(np.diag(pcov))
        if plot:
            plt.plot(xdata, ydata, 'b-', label='data')
            plt.plot(xdata, fit_function(xdata, *popt), 'r-', label='fit')
    except Exception as e:
        print(e)
        popt, perr = [], []
        if plot:
            plt.plot(xdata, ydata, 'b-', label='data')
    if plot:
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    return popt, perr


def multiple_fits(coefficients, exponents, Z):
    """Fit all orbitals in GTO basis with slater orbitals
    """

    def slater_2(r, a1, zeta1, zeta2):
        """dln(phi)/dr|r=r_nucl = -Z"""
        a2 = -a1
        return (
            a1 * np.exp(-zeta1*r)/(zeta1-Z) +
            a2 * np.exp(-zeta2*r)/(zeta2-Z)
        )

    def slater_3(r, a1, a2, zeta1, zeta2, zeta3):
        """dln(phi)/dr|r=r_nucl = -Z"""
        a3 = - (a1 + a2)
        return (
            a1 * np.exp(-zeta1*r)/(zeta1-Z) +
            a2 * np.exp(-zeta2*r)/(zeta2-Z) +
            a3 * np.exp(-zeta3*r)/(zeta3-Z)
        )

    def slater_4(r, a1, a2, a3, zeta1, zeta2, zeta3, zeta4):
        """dln(phi)/dr|r=r_nucl = -Z"""
        a4 = - (a1 + a2 + a3)
        return (
            a1 * np.exp(-zeta1*r)/(zeta1-Z) +
            a2 * np.exp(-zeta2*r)/(zeta2-Z) +
            a3 * np.exp(-zeta3*r)/(zeta3-Z) +
            a4 * np.exp(-zeta4*r)/(zeta4-Z)
        )

    def slater_5(r, *args):
        """dln(phi)/dr|r=r_nucl = -Z"""
        res = 0
        for coefficient, zeta in zip(coefficients, args):
            res += coefficient * np.exp(-zeta*r)/(zeta-Z)
        return Z * res

    def slater_6(r, *args):
        """dln(phi)/dr|r=r_nucl = -Z"""
        res = 0
        for coefficient, zeta in zip(coefficients, args):
            res += coefficient * np.exp(-zeta*r)/(zeta-Z)
        return Z * res

    fit_function = slater_2
    initial_guess = (1, Z-1, Z+1)

    # fit_function = slater_3
    # initial_guess = (1, 1, Z-1, Z+0.1, Z+1)

    # fit_function = slater_4
    # initial_guess = (1, 1, -1, Z-1, Z-1, Z+1, Z+1)

    # fit_function = slater_5
    # initial_guess = [1] * len(coefficients)

    xdata = np.linspace(0.01, 3.0, 50)
    ydata = wfn_s(xdata, coefficients, exponents)
    popt, perr = fit(fit_function, xdata, ydata, initial_guess, True)
    print(popt, perr)
    new_primitives = len(popt) // 2 + 1
    new_exponents = popt[new_primitives-1:]
    new_coefficients = np.append(popt[:new_primitives-1], -np.sum(popt[:new_primitives-1])) / (new_exponents - Z)
    print('dln(phi)/dr|r=r_nucl =', np.sum(new_coefficients * new_exponents/np.sum(new_coefficients)))
    return new_primitives, new_coefficients, new_exponents
