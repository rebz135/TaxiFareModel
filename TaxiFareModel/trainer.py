# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.pipeline import set_pipeline
from TaxiFareModel.mlflow import MLFlowBase
import joblib

class Trainer(MLFlowBase):
    def __init__(self):
        super().__init__(
            "[HK] rebz135 Taxi Fare Experiment LinearReg v1.0",
            "https://mlflow.lewagon.co/")

    def train(self):
        model_name = "linear"
        line_count = 1_000

        #log experiment
        self.mlflow_create_run()
        self.mlflow_log_param('estimator', model_name)
        self.mlflow_log_param("line_count", line_count)

        # get data
        df = get_data(line_count)

        # clean data
        df = clean_data(df)

        # set X and y
        y = df["fare_amount"]
        X = df.drop("fare_amount", axis=1)

        # hold out
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

        # build pipeline and train
        pipe = set_pipeline()
        pipe.fit(X_train, y_train)

        # evaluate and log
        y_pred = pipe.predict(X_val)
        rmse = compute_rmse(y_pred, y_val)
        self.mlflow_log_metric('rmse', rmse)
        print(rmse)

        # save model
        joblib.dump(pipe, 'test_save_model.joblib')

        return pipe

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    # loaded_model = joblib.load('test_save_model')
    # result = loaded_model.score(X_val, y_val)
    # print(result)
