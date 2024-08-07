import json
import datetime
import threading
import contextlib
import netCDF4 as nc
import numpy as np

from veros import (
    logger,
    variables,
    distributed,
    runtime_state,
    runtime_settings as rs,
    __version__ as veros_version,
)

"""
netCDF output is designed to follow the COARDS guidelines from
http://ferret.pmel.noaa.gov/Ferret/documentation/coards-netcdf-conventions
"""



def extract_init_cond(dataset, restart_vars, idx=0):
    initial_condition = {}
    for key in restart_vars.keys():
        if key == 'taum1':
            initial_condition['taum1'] = 0
        elif key == 'tau':
            initial_condition['tau'] = 1
        elif key == 'taup1':
            initial_condition['taup1'] = 2
        elif key == 'time':
            initial_condition['time'] = 0
        elif key == 'K_diss_v':
            initial_condition[key] = dataset[key][:].filled(fill_value=0.0)[idx - 1].T
        else:
            initial_condition[key] = dataset[key][:].filled(fill_value=0.0)[idx - 1:idx + 2].T
            initial_condition[key][..., -1] = 0 * initial_condition[key][..., -1]
    return initial_condition

def load_timesteps_between(file_path, start_timestep, end_timestep):
    """
    Load data between specified timesteps from all variables in a NetCDF file.

    Parameters:
    - file_path: str, path to the NetCDF file
    - start_timestep: int, the starting timestep (inclusive)
    - end_timestep: int, the ending timestep (exclusive)

    Returns:
    - data_chunks: dict, containing subsets of each variable
    """
    # Open the NetCDF file
    dataset = nc.Dataset(file_path, 'r')

    # Initialize a dictionary to store the subsets of each variable
    data_chunks = {}

    # Iterate over all variables in the dataset
    for var_name in dataset.variables:
        variable = dataset.variables[var_name]

        # Check if the variable has a time dimension (assuming the first dimension is time)
        if variable.ndim > 0 and variable.shape[0] >= end_timestep:
            # Load the data between the specified timesteps
            data_chunks[var_name] = variable[start_timestep:end_timestep, ...]
        else:
            # If the variable does not have a time dimension, load it completely
            data_chunks[var_name] = variable[...]

    # Close the dataset
    dataset.close()

    return data_chunks



def _get_setup_code(pyfile):
    try:
        with open(pyfile, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "UNKNOWN"


def initialize_file(state, ncfile, extra_dimensions=None, create_time_dimension=True, include_ghosts=False):
    """
    Define standard grid in netcdf file
    """
    import h5netcdf

    if not isinstance(ncfile, h5netcdf.File):
        raise TypeError("Argument needs to be a netCDF4 Dataset")

    if rs.setup_file is None:
        setup_file = "UNKNOWN"
        setup_code = "UNKNOWN"
    else:
        setup_file = rs.setup_file
        setup_code = _get_setup_code(rs.setup_file)

    ncfile.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        veros_version=veros_version,
        setup_identifier=state.settings.identifier,
        setup_description=state.settings.description,
        setup_settings=json.dumps(state.settings.todict()),
        setup_file=setup_file,
        setup_code=setup_code,
    )

    dimensions = dict(state.dimensions)
    if extra_dimensions is not None:
        dimensions.update(extra_dimensions)

    for dim in dimensions:
        # time steps are peeled off explicitly
        if dim in variables.TIMESTEPS:
            continue

        if dim in state.var_meta:
            var = state.var_meta[dim]

            # skip inactive dimensions
            if not var.active:
                continue

            var_data = getattr(state.variables, dim)
        else:
            # create dummy variable for dimensions without data
            var = variables.Variable(dim, (dim,), time_dependent=False)
            var_data = np.arange(dimensions[dim])

        dimsize = variables.get_shape(dimensions, var.dims[::-1], include_ghosts=include_ghosts, local=False)[0]
        ncfile.dimensions[dim] = dimsize
        initialize_variable(state, dim, var, ncfile, include_ghosts=include_ghosts)
        write_variable(state, dim, var, var_data, ncfile, include_ghosts=include_ghosts)

    if create_time_dimension:
        ncfile.dimensions["Time"] = None
        nc_dim_var_time = ncfile.create_variable("Time", ("Time",), float)
        nc_dim_var_time.attrs.update(
            long_name="Time",
            units="days",
            time_origin="01-JAN-1900 00:00:00",
        )


def initialize_variable(state, key, var, ncfile, include_ghosts=False):
    if var.dims is None:
        dims = ()
    else:
        dims = tuple(d for d in var.dims if d in ncfile.dimensions)

    if var.time_dependent and "Time" in ncfile.dimensions:
        dims += ("Time",)

    if key in ncfile.variables:
        logger.warning(f"Variable {key} already initialized")
        return

    kwargs = {}
    if rs.hdf5_gzip_compression and runtime_state.proc_num == 1:
        kwargs.update(compression="gzip", compression_opts=1)

    chunksize = [
        variables.get_shape(state.dimensions, (d,), local=True, include_ghosts=include_ghosts)[0] if d in state.dimensions else 1
        for d in dims
    ]

    dtype = var.dtype
    if dtype is None:
        dtype = rs.float_type
    elif dtype == "bool":
        dtype = "uint8"

    fillvalue = variables.get_fill_value(dtype)

    # transpose all dimensions in netCDF output (convention in most ocean models)
    v = ncfile.create_variable(key, dims[::-1], dtype, fillvalue=fillvalue, chunks=tuple(chunksize[::-1]), **kwargs)
    v.missing_value = fillvalue
    v.attrs.update(long_name=var.name, units=var.units, **var.extra_attributes)


def advance_time(time_value, ncfile):
    current_time_step = len(ncfile.variables["Time"])
    ncfile.resize_dimension("Time", current_time_step + 1)
    ncfile.variables["Time"][current_time_step] = time_value


def add_dimension(dim, dim_size, ncfile):
    ncfile.dimensions[dim] = int(dim_size)


def write_variable(state, key, var, var_data, ncfile, time_step=-1, include_ghosts=False):
    var_data = var_data * var.scale

    gridmask = var.get_mask(state.settings, state.variables)
    if gridmask is not None:
        newaxes = (slice(None),) * gridmask.ndim + (np.newaxis,) * (var_data.ndim - gridmask.ndim)
        var_data = np.where(gridmask.astype("bool")[newaxes], var_data, variables.get_fill_value(var_data.dtype))

    if var.dims:
        tmask = tuple(state.variables.tau if dim in variables.TIMESTEPS else slice(None) for dim in var.dims)
        if not include_ghosts:
            var_data = variables.remove_ghosts(var_data, var.dims)[tmask].T
        else:
            var_data = var_data[tmask].T

    var_obj = ncfile.variables[key]
    nx, ny = state.dimensions["xt"], state.dimensions["yt"]
    if include_ghosts:
        nx, ny = nx + 4, ny + 4
    chunk, _ = distributed.get_chunk_slices(nx, ny, var_obj.dimensions)

    if "Time" in var_obj.dimensions:
        assert var_obj.dimensions[0] == "Time"
        chunk = (time_step,) + chunk[1:]

    var_obj[chunk] = var_data


@contextlib.contextmanager
def threaded_io(filepath, mode):
    """
    If using IO threads, start a new thread to write the netCDF data to disk.
    """
    import h5py
    import h5netcdf

    if rs.use_io_threads:
        _wait_for_disk(filepath)
        _io_locks[filepath].clear()

    kwargs = dict()

    if int(h5py.__version__.split(".")[0]) >= 3:
        kwargs.update(decode_vlen_strings=True)

    if runtime_state.proc_num > 1:
        kwargs.update(driver="mpio", comm=rs.mpi_comm)

    nc_dataset = h5netcdf.File(filepath, mode, **kwargs)

    try:
        yield nc_dataset

    finally:
        if rs.use_io_threads:
            threading.Thread(target=_write_to_disk, args=(nc_dataset, filepath)).start()
        else:
            _write_to_disk(nc_dataset, filepath)


_io_locks = {}


def _add_to_locks(file_id):
    """
    If there is no lock for file_id, create one
    """
    if file_id not in _io_locks:
        _io_locks[file_id] = threading.Event()
        _io_locks[file_id].set()


def _wait_for_disk(file_id):
    """
    Wait for the lock of file_id to be released
    """
    logger.debug(f"Waiting for lock {file_id} to be released")
    _add_to_locks(file_id)
    lock_released = _io_locks[file_id].wait(rs.io_timeout)

    if not lock_released:
        raise RuntimeError("Timeout while waiting for disk IO to finish")


def _write_to_disk(ncfile, file_id):
    """
    Sync netCDF data to disk, close file handle, and release lock.
    May run in a separate thread.
    """
    try:
        ncfile.close()
    finally:
        if rs.use_io_threads and file_id is not None:
            _io_locks[file_id].set()
