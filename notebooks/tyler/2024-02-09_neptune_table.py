##
import neptune.new as neptune
##
project = neptune.init_project(project="neuro/gaddy")
ex_run = project.fetch_run("GAD-960")
##
runs_table_df = project.fetch_runs_table(columns=["training/hyperparams/batch_size"]).to_pandas()
runs_table_df
##
runs_table_df = project.fetch_runs_table(
    columns=[
        "training/hyperparams/batch_size",
        "training/val/wer"
             ]
).to_pandas()
runs_table_df
##
project["training/val/wer"].fetch()
##
