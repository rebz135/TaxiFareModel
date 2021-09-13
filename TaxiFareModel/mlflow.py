import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property


class MLFlowBase():
    def __init__(self, experiment_name='rebz135 taxi fare', MLFLOW_URI="https://mlflow.lewagon.co"):
        self.experiment_name = experiment_name
        self.MLFLOW_URI = MLFLOW_URI

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    def mlflow_create_run(self):
        self.mlflow_run = self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
