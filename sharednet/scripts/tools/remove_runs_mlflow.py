import mlflow
import shutil


def get_run_dir(artifacts_uri):
    return artifacts_uri[7:-10]


def remove_run_dir(run_dir):
    shutil.rmtree(run_dir, ignore_errors=True)


experiment_id = 0  # default experiment ID = 0
deleted_runs = 2  # active, deleted, all = 1, 2, 3, respectively

exp = mlflow.tracking.MlflowClient(tracking_uri='../mlruns')  # path of mlruns

runs = exp.search_runs(str(experiment_id), run_view_type=deleted_runs)

for run in runs:
    # print(run.info.artifact_uri)
    deleted_dir = get_run_dir(run.info.artifact_uri)
    print(deleted_dir)
    remove_run_dir(deleted_dir)

# _ = [remove_run_dir(get_run_dir(run.info.artifact_uri)) for run in runs]