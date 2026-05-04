import numpy as np

class Viscosity:

    # 21. crude oil viscosity
    # 22. methods of calculating the dead oil viscosity
    # 23. methods of calculating the saturated oil viscosity
    # 24. methods of calculating the viscosity of the undersaturated oil
    def oil(temp, p, Tsep, Psep, bpp, Rs, sgsg, gAPI):
        """
        Calculates the Oil Viscosity in cp

        temp   : temperature, °F
        p      : pressure, psia
        Tsep   : separator temperature, °F
        Psep   : separator pressure, psia
        bpp    : bubble point pressure, psia
        gass   : solution gas-oil ratio, scf/stb
        sgsg   : gas specific gravity
        gAPI   : API oil gravity

        """
        a = 10.715 * (gass + 100) ** (-0.515)
        b = 5.44 * (gass + 150) ** (-0.338)
        Z = 3.0324 - 0.0203 * gAPI
        Y = 10 **Z
        x = Y * temp ** (-1.163)

        visc_oD = 10 ** x - 1

        if (p <= bpp):
            visc_o = a * visc_oD ** b
        else:
            M = 2.6 * p ** 1.187 * np.exp(-11.513 - 8.98E-05 * p)
            visc_ob = a * visc_oD ** b
            visc_o = visc_ob * (p / bpp) ** M

        return visc_o