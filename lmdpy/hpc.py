
"""
lmdpy.hpc

Helpers to run large ensembles of Langevin simulations on an HPC cluster.

This module does not depend on any specific scheduler, but it provides:

    - A small multiprocessing-based ensemble runner for local machines.
    - Template generation for HTCondor submit files.
"""

import os
import textwrap
import multiprocessing as mp


def run_ensemble_local(simulation_fn, param_list, max_workers=None):
    """
    Run many simulations in parallel using multiprocessing.

    Parameters
    ----------
    simulation_fn : callable(param) -> result
        Function that takes a single parameter object and returns a result dict
        (for example, a wrapper around LangevinIntegrator.simulate()).
    param_list : list
        List of parameter objects to iterate over.
    max_workers : int or None
        Number of worker processes. Defaults to mp.cpu_count().

    Returns
    -------
    results : list
        List of results in the same order as param_list.
    """
    if max_workers is None:
        max_workers = mp.cpu_count()

    with mp.Pool(processes=max_workers) as pool:
        results = pool.map(simulation_fn, param_list)

    return results


def htcondor_submit_template(
    executable,
    arguments="",
    log_dir="logs",
    output_dir="out",
    error_dir="err",
    n_jobs=1,
):
    """
    Generate a string containing a minimal HTCondor submit file.

    Parameters
    ----------
    executable : str
        Path to the Python executable or wrapper script.
    arguments : str
        Command-line arguments (can contain $(Process) for indexing).
    log_dir, output_dir, error_dir : str
        Directories for HTCondor logs, stdout, stderr.
    n_jobs : int
        Number of jobs to queue.

    Returns
    -------
    submit_str : str
    """
    return textwrap.dedent(
        f"""
        universe                = vanilla
        executable              = {executable}
        arguments               = {arguments}

        log                     = {log_dir}/job_$(Cluster)_$(Process).log
        output                  = {output_dir}/job_$(Cluster)_$(Process).out
        error                   = {error_dir}/job_$(Cluster)_$(Process).err

        request_cpus            = 1
        request_memory          = 2GB

        should_transfer_files   = YES
        when_to_transfer_output = ON_EXIT

        queue {n_jobs}
        """
    ).strip()


def write_htcondor_submit_file(path, *args, **kwargs):
    """
    Convenience wrapper around htcondor_submit_template that writes to disk.
    """
    content = htcondor_submit_template(*args, **kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path
