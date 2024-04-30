import torch
import numpy as np

class MakeInputDatas:
    def __init__(self, Len, Period):
        self.Len = Len
        self.Period = Period
        self.Cycl = Len / Period
        self.Vp = 220
        self.w = 2 * np.pi / Period
        self.n_t = np.arange(1, Len + 1)
        self.regen()

    def add_noise(self, sig, rate):
        rms = np.sqrt(np.mean(sig**2))
        noise = (2 * np.random.rand(len(sig)) - 1) * rms * rate
        return sig + noise

    def regen(self):
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

    def current(self, v, dvdn, g, c):
        return g * v + c * dvdn

    def get_data(self):
        data = []
        data.append(torch.tensor([
            self.V1_t,
            self.V2_t,
            self.V3_t,
            self.dV1dn_t,
            self.dV2dn_t,
            self.dV3dn_t,
            self.I_t
        ]).float().flatten())
        data = torch.stack(data)
        return data
    
    def get_target(self):
        return torch.tensor(torch.tensor([
            self.G1_t,
            self.G2_t,
            self.G3_t,
            self.C1_t,
            self.C2_t,
            self.C3_t,
        ]).float().flatten())
