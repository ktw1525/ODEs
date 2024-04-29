import numpy as np

class MakeInputDatas:
    def __init__(self, Len, Period):
        self.Len = Len
        self.Period = Period
        self.Cycl = Len / Period
        self.Vp = 220
        self.w = 2 * np.pi / Period
        self.n_t = np.arange(1, Len + 1)

        # Coefficients initialization
        self.G1_t = 0.0002 * np.random.rand()
        self.C1_t = 0.0002 * np.random.rand()
        self.G2_t = 0.0002 * np.random.rand()
        self.C2_t = 0.0002 * np.random.rand()
        self.G3_t = 0.0002 * np.random.rand()
        self.C3_t = 0.0002 * np.random.rand()

        # Generate and add noise to voltages
        self.V1_t = self.add_noise(self.Vp * np.sin(self.w * self.n_t), 0.1)
        self.V2_t = self.add_noise(self.Vp * np.sin(self.w * self.n_t + 2 * np.pi / 3), 0.1)
        self.V3_t = self.add_noise(self.Vp * np.sin(self.w * self.n_t - 2 * np.pi / 3), 0.1)

        # Derivatives of voltages
        self.dV1dn_t = np.append([0], np.diff(self.V1_t))
        self.dV2dn_t = np.append([0], np.diff(self.V2_t))
        self.dV3dn_t = np.append([0], np.diff(self.V3_t))

        # Current calculations
        self.I_t = self.current(self.V1_t, self.dV1dn_t, self.G1_t, self.C1_t) + \
                   self.current(self.V2_t, self.dV2dn_t, self.G2_t, self.C2_t) + \
                   self.current(self.V3_t, self.dV3dn_t, self.G3_t, self.C3_t)
        self.I_t = self.add_noise(self.I_t, 0.00)

    def add_noise(self, sig, rate):
        rms = np.sqrt(np.mean(sig**2))
        noise = (2 * np.random.rand(len(sig)) - 1) * rms * rate
        return sig + noise

    def current(self, v, dvdn, g, c):
        return g * v + c * dvdn

    def get_data(self):
        return {
            "n": self.n_t,
            "V1": self.V1_t,
            "V2": self.V2_t,
            "V3": self.V3_t,
            "dV1dn": self.dV1dn_t,
            "dV2dn": self.dV2dn_t,
            "dV3dn": self.dV3dn_t,
            "I": self.I_t,
            "G1": self.G1_t,
            "G2": self.G2_t,
            "G3": self.G3_t,
            "C1": self.C1_t,
            "C2": self.C2_t,
            "C3": self.C3_t
        }
