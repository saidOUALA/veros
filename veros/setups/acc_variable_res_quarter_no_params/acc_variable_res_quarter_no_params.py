#!/usr/bin/env python

"""
This Veros setup file was generated by

   $ veros copy-setup acc

on 2024-03-29 08:50:01 UTC.
"""

__VEROS_VERSION__ = '0+untagged.1774.g4039f76.dirty'

if __name__ == "__main__":
    raise RuntimeError(
        "Veros setups cannot be executed directly. "
        f"Try `veros run {__file__}` instead."
    )

# -- end of auto-generated header, original file below --


from veros import VerosSetup, veros_routine
from veros.variables import allocate, Variable
from veros.distributed import global_min, global_max
from veros.core.operators import numpy as npx, update, at


class ACCResQNPSetup(VerosSetup):
    """A model using spherical coordinates with a partially closed domain representing the Atlantic and ACC.

    Wind forcing over the channel part and buoyancy relaxation drive a large-scale meridional overturning circulation.

    This setup demonstrates:
     - setting up an idealized geometry
     - updating surface forcings
     - basic usage of diagnostics

    `Adapted from pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/ACC%202>`_.
    """

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        settings.identifier = "acc_runs/acc_simulation_quarter_no_param/acc_simulation_quarter_no_param_post_spinup_viz"
        settings.description = "My ACC setup"
        settings.restart_input_filename = "acc_runs/acc_simulation_quarter_no_param/acc_simulation_quarter_no_param_post_spinup_50000.restart.h5"

        nb_years = 4
        seconds_per_year = 31557600
        res = 1/4
        delta = 2/res
        ratio = delta**2
        settings.nx, settings.ny, settings.nz = 248,324,15#,30, 42, 15
        settings.dt_mom = 4800/delta
        settings.dt_tracer = 4800/delta
        settings.runlen = nb_years * seconds_per_year#/(settings.dt_tracer/delta) #delta*100000 * settings.dt_tracer*2/3

        settings.x_origin = 0.0
        settings.y_origin = -40.0

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        # coefs for isopycnal tracer diffusion 
        settings.enable_neutral_diffusion = False#True
        settings.K_iso_0 = 1000.0/ratio
        settings.K_iso_steep = 500.0/ratio
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.01
        settings.enable_skew_diffusion = False#True

        settings.enable_hor_friction = True
        settings.A_h = (2 * settings.degtom) ** 3 * 2e-11/ratio
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1

        settings.enable_bottom_friction = True
        settings.r_bot = 1e-5

        settings.enable_implicit_vert_friction = True

        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True

        settings.K_gm_0 = 1000.0/ratio
        settings.enable_eke = True
        settings.eke_k_max = 1e4/ratio
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.0
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0/100
        settings.enable_eke_superbee_advection = True
        settings.enable_eke_isopycnal_diffusion = True

        settings.enable_idemix = False

        settings.eq_of_state_type = 3

        var_meta = state.var_meta
        var_meta.update(
            t_star=Variable("t_star", ("yt",), "deg C", "Reference surface temperature"),
            t_rest=Variable("t_rest", ("xt", "yt"), "1/s", "Surface temperature restoring time scale"),
        )

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        ddz = npx.array(
            [50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0]
        )
        vs.dxt = update(vs.dxt, at[...], 1/4)
        vs.dyt = update(vs.dyt, at[...], 1/4)
        vs.dzt = update(vs.dzt, at[...], ddz[::-1] / 2.5)

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(
            vs.coriolis_t, at[...], 2 * settings.omega * npx.sin(vs.yt[None, :] / 180.0 * settings.pi)
        )

    @veros_routine
    def set_topography(self, state):
        vs = state.variables
        x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")
        vs.kbot = npx.logical_or(x > 1.0, y < -20).astype("int")

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        # initial conditions
        vs.temp = update(vs.temp, at[...], ((1 - vs.zt[None, None, :] / vs.zw[0]) * 15 * vs.maskT)[..., None])
        vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., None])

        # wind stress forcing
        yt_min = global_min(vs.yt.min())
        yu_min = global_min(vs.yu.min())
        yt_max = global_max(vs.yt.max())
        yu_max = global_max(vs.yu.max())

        taux = allocate(state.dimensions, ("yt",))
        taux = npx.where(vs.yt < -20, 0.1 * npx.sin(settings.pi * (vs.yu - yu_min) / (-20.0 - yt_min)), taux)
        taux = npx.where(vs.yt > 10, 0.1 * (1 - npx.cos(2 * settings.pi * (vs.yu - 10.0) / (yu_max - 10.0))), taux)
        vs.surface_taux = taux * vs.maskU[:, :, -1]

        # surface heatflux forcing
        vs.t_star = allocate(state.dimensions, ("yt",), fill=15)
        vs.t_star = npx.where(vs.yt < -20, 15 * (vs.yt - yt_min) / (-20 - yt_min), vs.t_star)
        vs.t_star = npx.where(vs.yt > 20, 15 * (1 - (vs.yt - 20) / (yt_max - 20)), vs.t_star)
        vs.t_rest = vs.dzt[npx.newaxis, -1] / (30.0 * 86400.0) * vs.maskT[:, :, -1]

        if settings.enable_tke:
            vs.forc_tke_surface = update(
                vs.forc_tke_surface,
                at[2:-2, 2:-2],
                npx.sqrt(
                    (0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / settings.rho_0) ** 2
                    + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / settings.rho_0) ** 2
                )
                ** (1.5),
            )

        if settings.enable_idemix:
            vs.forc_iw_bottom = 1e-6 * vs.maskW[:, :, -1]
            vs.forc_iw_surface = 1e-7 * vs.maskW[:, :, -1]

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.forc_temp_surface = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        diagnostics = state.diagnostics

        diagnostics["acc_monitor"].output_frequency = settings.dt_tracer
        diagnostics["averages"].output_variables = (
            "salt",
            "temp",
            "u",
            "v",
            "w",
            "psi",
            "surface_taux",
            "surface_tauy",
        )
        diagnostics["averages"].output_frequency = 365 * 86400.0
        diagnostics["averages"].sampling_frequency = settings.dt_tracer
        #diagnostics["overturning"].output_frequency = 365 * 86400.0 / 48.0
        #diagnostics["overturning"].sampling_frequency = settings.dt_tracer * 10
        #diagnostics["tracer_monitor"].output_frequency = 365 * 86400.0 / 12.0
        diagnostics["energy"].output_frequency   = settings.dt_tracer
        diagnostics["energy"].sampling_frequency = settings.dt_tracer

    @veros_routine
    def after_timestep(self, state):
        pass
