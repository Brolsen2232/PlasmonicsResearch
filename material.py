import meep as mp

class Material:
    def __init__(self, material_type='gold'):
        self.material_type = material_type
    
    def make_material(self):
        if self.material_type == 'gold':
            return self.make_gold_material()
        else:
            raise ValueError("Unsupported material type")
    
    def make_gold_material():
        metal_range = mp.FreqRange(min=um_scale / 6.1992, max=um_scale / 0.24797)
        Au_plasma_frq = 9.03 * eV_um_scale
        Au_f0 = 0.760
        Au_frq0 = 1e-10
        Au_gam0 = 0.053 * eV_um_scale
        Au_sig0 = Au_f0 * Au_plasma_frq**2 / Au_frq0**2
        Au_f1 = 0.024
        Au_frq1 = 0.415 * eV_um_scale  # 2.988 μm
        Au_gam1 = 0.241 * eV_um_scale
        Au_sig1 = Au_f1 * Au_plasma_frq**2 / Au_frq1**2
        Au_f2 = 0.010
        Au_frq2 = 0.830 * eV_um_scale  # 1.494 μm
        Au_gam2 = 0.345 * eV_um_scale
        Au_sig2 = Au_f2 * Au_plasma_frq**2 / Au_frq2**2
        Au_f3 = 0.071
        Au_frq3 = 2.969 * eV_um_scale  # 0.418 μm
        Au_gam3 = 0.870 * eV_um_scale
        Au_sig3 = Au_f3 * Au_plasma_frq**2 / Au_frq3**2
        Au_f4 = 0.601
        Au_frq4 = 4.304 * eV_um_scale  # 0.288 μm
        Au_gam4 = 2.494 * eV_um_scale
        Au_sig4 = Au_f4 * Au_plasma_frq**2 / Au_frq4**2
        Au_f5 = 4.384
        Au_frq5 = 13.32 * eV_um_scale  # 0.093 μm
        Au_gam5 = 2.214 * eV_um_scale
        Au_sig5 = Au_f5 * Au_plasma_frq**2 / Au_frq5**2

        Au_susc = [
            mp.DrudeSusceptibility(frequency=Au_frq0, gamma=Au_gam0, sigma=Au_sig0),
            mp.LorentzianSusceptibility(frequency=Au_frq1, gamma=Au_gam1, sigma=Au_sig1),
            mp.LorentzianSusceptibility(frequency=Au_frq2, gamma=Au_gam2, sigma=Au_sig2),
            mp.LorentzianSusceptibility(frequency=Au_frq3, gamma=Au_gam3, sigma=Au_sig3),
            mp.LorentzianSusceptibility(frequency=Au_frq4, gamma=Au_gam4, sigma=Au_sig4),
            mp.LorentzianSusceptibility(frequency=Au_frq5, gamma=Au_gam5, sigma=Au_sig5),
        ]

        return mp.Medium(epsilon=1.0, E_susceptibilities=Au_susc, valid_freq_range=metal_range)


