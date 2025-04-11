"""Generate slurm script for experiment using settings & schedule."""

import os
import sys
import subprocess

# NOTE: could expand this script to parse more slurm args, e.g. username etc.,
# and provide a generic template, but this'll do for now

if __name__ == "__main__":

    # Get experiment settings
    expt_name = str(sys.argv[1])
    args = " ".join(str(a) for a in sys.argv[2:])
    expt_no = args.replace(" ", "")

    assert expt_name in ['prior','post_design','post_eval'], f"Experiment type must be one of 'prior','post_design','post_eval', {expt_name} given."

    # set up experiment settings
    if expt_name == 'prior':
        expt_long_name = 'find_and_eval_prior_systems'
        partition = 'cclake-himem'
        n_jobs = 4
        n_cpus = 10
        walltime = '12:00:00'
        nodecancel = ""
    elif expt_name == 'post_design':
        expt_long_name = 'find_posterior_solns'
        partition = "cclake-himem"
        n_jobs = 255
        n_cpus = 4
        walltime = '04:00:00'
        nodecancel = ""
    elif expt_name == 'post_eval':
        expt_long_name = 'eval_posterior_solns'
        partition = 'cclake'
        n_jobs = 255
        n_cpus = 10
        walltime = '04:00:00'
        nodecancel = "#"

    # create slurm script for experiment
    slurm_template = 'slurm_submit_cc_template'
    script_name = 'ssub_cc_temp'
    with open(slurm_template,'r') as fread:
        text = fread.read()
    text = text.format(
        expt_name=expt_name,
        expt_long_name=expt_long_name,
        expt_no=expt_no,
        args=args,
        partition=partition,
        walltime=walltime,
        nodecancel=nodecancel,
        n_cpus=n_cpus,
        n_jobs=n_jobs
    )
    with open(script_name,'w') as fout:
        fout.write(text)

    # schedule job using sbatch
    p = subprocess.run(f'sbatch {script_name}',
                       shell=True,
                       check=True,
                       capture_output=True,
                       encoding='utf-8'
                       )
    print(f'Command {p.args} exited with {p.returncode} code\nOutput:\n{p.stdout}')

    # clean up shell script
    os.remove(script_name)

    print(
        "Successfully scheduled jobs with following options:\n"\
        f"Experiment name: {expt_name}\n"\
        f"Experiment no: {expt_no}\n"\
        f"Walltime: {walltime}\n"\
        f"CPUs per task: {n_cpus}\n"\
        f"No. jobs in array: {n_jobs}\n"
    )