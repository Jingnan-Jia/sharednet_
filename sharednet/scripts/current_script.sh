#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
##SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH -e results/logs/slurm-%j.err
#SBATCH -o results/logs/slurm-%j.out
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com


eval "$(conda shell.bash hook)"

conda activate py38

job_id=$SLURM_JOB_ID
slurm_dir=results/logs
echo job_id is $job_id
##cp script.sh ${slurm_dir}/slurm-${job_id}.shs
# git will not detect the current file because this file may be changed when this job was run
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh

# quote ENDSSH to make sure echo $jobs is not empty
# https://stackoverflow.com/questions/34567748/bash-variable-is-always-empty
# The following code will ssh to loginnode and git commit to synchronize commits from different nodes.

# But sleep some time is required otherwise multiple commits by several experiments at the same time
# will lead to commit error: fatal: could not parse HEAD


ssh -tt jjia@nodelogin02 /bin/bash << ENDSSH
echo "Hello, I an in nodelogin02 to do some git operations."
echo $job_id

jobs=$(squeue -u jjia --sort=+i | grep [^0-9]0:[00-60] | awk '{print $1}')
echo "Total jobs in one minutes:"
echo \$jobs

accu=0
for i in \$jobs; do
    if [[ \$i -eq $job_id ]]; then
    echo start sleep ...
    sleep \$accu
    echo sleep \$accu seconds
    fi

    accu=$(( $accu+5 ))
done

cd data/sharednet
echo $job_id
scontrol write batch_script "${job_id}" sharednet/scripts/current_script.sh  # for the git commit latter

git add -A
git commit -m "jobid is ${job_id}"
git push origin master
exit
ENDSSH

echo "Hello, I am back in $(hostname) to run the code"

# shellcheck disable=SC2046
idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_${idx}_err.txt 1>${slurm_dir}/slurm-${job_id}_${idx}_out.txt --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --jobid=${job_id} --model_names="liver" --cond_flag=False --cond_pos='enc_dec' --mode='train' --infer_ID=0 --remark="pancreas"




