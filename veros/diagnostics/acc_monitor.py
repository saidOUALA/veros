from veros import logger
from veros import logger, veros_kernel, KernelOutput
import os
from veros.variables import Variable
from veros.core.operators import numpy as npx
from veros.diagnostics.base import VerosDiagnostic
from veros.distributed import global_sum
from veros.core.operators import numpy as npx, update, update_add, at, for_loop



ACC_DIAG_VARIABLES = dict(
    nitts=Variable("nitts", None, write_to_restart=True),
    # mean energy content
    temp_zonal_mean=Variable("temp_zonal_mean", ("yu", "zw"),  write_to_restart=True),
    salt_zonal_mean=Variable("salt_zonal_mean", ("yu", "zw"), write_to_restart=True),
    flux_north_south=Variable("flux_north_south", ("xu",),  write_to_restart=True),
)

DEFAULT_OUTPUT_VARS = [var for var in ACC_DIAG_VARIABLES.keys() if var not in ("nitts",)]


class ACCMonitor(VerosDiagnostic):
    """Diagnostic monitoring global tracer contents / fluxes.

    Writes output to stdout (no binary output).
    """

    name = "acc_monitor"
    output_path = "{identifier}.acc_diags.nc"  #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.

    var_meta = ACC_DIAG_VARIABLES
    def __init__(self, state):
        self.output_variables = DEFAULT_OUTPUT_VARS.copy()


    def initialize(self, state):
        self.initialize_variables(state)

    def diagnose(self, state):
        ovt_vs = self.variables
        ovt_vs.update(diagnose_kernel(state, ovt_vs))
        self.variables.nitts = self.variables.nitts + 1

    def output(self, state):
        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)
        nitts = float(self.variables.nitts or 1)
        self.write_output(state)
        self.variables.nitts = 0
def compute_flux(velocity_field, delta_y, delta_z):
    Nx = velocity_field.shape[0]

    # Initialize an array to store flux values for each section
    fluxes = npx.zeros(Nx)
    yy, zz = npx.meshgrid(delta_y, delta_z, indexing='ij')
    for section_index in range(Nx):
        # Extract the v_x component of the velocity field at the given section index
        v_x_section = velocity_field[section_index, :, :]

        # Compute the flux for the current section
        fluxes[section_index] = npx.sum(v_x_section * yy * zz)

    return fluxes

@veros_kernel
def diagnose_kernel(state, ovt_vs):
    vs = state.variables
    settings = state.settings

    temp_zonal_mean_tmp = state.variables.temp[2:-2, 2:-2, :, state.variables.tau].mean(axis=0)
    salt_zonal_mean_tmp = state.variables.salt[2:-2, 2:-2, :, state.variables.tau].mean(axis=0)
    flux_north_south_tmp = compute_flux(state.variables.u[2:-2, 2:-2, :, state.variables.tau], state.variables.dyu[2:-2], state.variables.dzt)

    ovt_vs.temp_zonal_mean = update(
        ovt_vs.temp_zonal_mean,
        at[2:-2, :],temp_zonal_mean_tmp)

    ovt_vs.salt_zonal_mean = update(
        ovt_vs.salt_zonal_mean,
        at[2:-2, :],salt_zonal_mean_tmp)

    ovt_vs.flux_north_south = update(
        ovt_vs.flux_north_south,
        at[2:-2],flux_north_south_tmp)

    return KernelOutput(
        temp_zonal_mean=ovt_vs.temp_zonal_mean,
        salt_zonal_mean=ovt_vs.salt_zonal_mean,
        flux_north_south=ovt_vs.flux_north_south,
    )
