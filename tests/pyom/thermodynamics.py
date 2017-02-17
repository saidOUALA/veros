from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from pyomtest import PyOMTest
from climate import Timer
from climate.pyom import thermodynamics

class ThermodynamicsTest(PyOMTest):
    extra_settings = {
                        "enable_cyclic_x": True,
                        "enable_conserve_energy": True,
                        "enable_hor_friction_cos_scaling": True,
                        "enable_tempsalt_sources": True,
                        "enable_hor_diffusion": True,
                        "enable_superbee_advection": True,
                        "enable_tke": True,
                        "enable_biharmonic_mixing": True,
                        "enable_neutral_diffusion": True,
                        "enable_skew_diffusion": True,
                     }
    def initialize(self):
        m = self.pyom_legacy.main_module

        self.set_attribute("hor_friction_cosPower", np.random.randint(1,5))

        for a in ("iso_slopec","iso_dslope","K_iso_steep","dt_tracer","dt_mom","K_hbi","K_h","AB_eps"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt","dxu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.nx+4).astype(np.float))

        for a in ("dyt","dyu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.ny+4).astype(np.float))

        for a in ("cosu","cost"):
            self.set_attribute(a,2*np.random.rand(self.ny+4)-1.)

        for a in ("zt","dzt","dzw"):
            self.set_attribute(a, np.random.rand(self.nz))

        for a in ("area_u", "area_v", "area_t", "forc_rho_surface", "forc_temp_surface"):
            self.set_attribute(a, np.random.rand(self.nx+4, self.ny+4))

        for a in ("flux_east","flux_north","flux_top","dtemp_hmix","dsalt_hmix","temp_source",
                  "salt_source","u_wgrid","v_wgrid","w_wgrid","K_iso","K_gm","kappa_gm","du_mix",
                  "P_diss_iso","P_diss_skew","P_diss_v","P_diss_nonlin",
                  "kappaH"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("Hd","dHd","temp","salt","int_drhodS","int_drhodT","dtemp","dsalt",
                  "u","v","w","Nsqr","tke"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a,np.random.randint(0,2,size=(self.nx+4,self.ny+4,self.nz)).astype(np.float))

        self.set_attribute("kbot",np.random.randint(0, self.nz, size=(self.nx+4,self.ny+4)))

        self.test_module = thermodynamics
        pyom_args = (self.pyom_new,)
        pyom_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines.update(thermodynamics = (pyom_args, pyom_legacy_args),)

    def test_passed(self,routine):
        all_passed = True
        for f in ("flux_east","flux_north","flux_top","temp","salt",
                  "dtemp","dsalt","P_diss_iso","Hd","dHd","Nsqr","P_diss_adv",
                  "dtemp_iso", "dsalt_iso","dtemp_vmix","dsalt_vmix","forc_rho_surface",
                  "P_diss_v","P_diss_nonlin","int_drhodT","int_drhodS","rho"):
            passed = self._check_var(f)
            if not passed:
                all_passed = False
        plt.show()
        return all_passed

    def _normalize(self,*arrays):
        norm = np.abs(arrays[0]).max()
        if norm == 0.:
            return arrays
        return (a / norm for a in arrays)

    def _check_var(self,var):
        v1, v2 = self.get_attribute(var)
        if v1.ndim > 1:
            v1 = v1[2:-2, 2:-2, ...]
        if v2.ndim > 1:
            v2 = v2[2:-2, 2:-2, ...]
        if v1 is None or v2 is None:
            raise RuntimeError(var)
        passed = np.allclose(*self._normalize(v1,v2))
        if not passed:
            print(var, np.abs(v1-v2).max(), v1.max(), v2.max(), np.where(v1 != v2))
            while v1.ndim > 2:
                v1 = v1[...,-1]
            while v2.ndim > 2:
                v2 = v2[...,-1]
            if v1.ndim == 2:
                fig, axes = plt.subplots(1,3)
                axes[0].imshow(v1)
                axes[0].set_title("New")
                axes[1].imshow(v2)
                axes[1].set_title("Legacy")
                axes[2].imshow(v1 - v2)
                axes[2].set_title("diff")
                fig.suptitle(var)
        return passed

if __name__ == "__main__":
    test = ThermodynamicsTest(70, 60, 50, fortran=sys.argv[1])
    passed = test.run()
