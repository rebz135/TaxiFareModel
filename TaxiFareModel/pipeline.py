from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder


def set_pipeline(estimator=('linear_model', LinearRegression())):
    """defines the pipeline as a class attribute"""
    dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                          ('stdscaler', StandardScaler())])

    time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))])

    preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
        "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
        'dropoff_longitude'
    ]), ('time', time_pipe, ['pickup_datetime'])],
                                     remainder="drop")

    pipe = Pipeline([('preproc', preproc_pipe), estimator])

    return pipe
