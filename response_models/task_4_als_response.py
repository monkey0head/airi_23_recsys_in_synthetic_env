from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame

import pyspark.sql.functions as sf

from sim4rec.response import BernoulliResponse, ActionModelTransformer

from response_models.utils import get_session, ResponseTransformer
from pyspark.sql import SparkSession


class TaskFourResponse:
    def __init__(self,
                 spark,
                 als_df_path='./response_models/data/als_response_df_1000_items.parquet',
                 seed=123):
        als_resp = ResponseTransformer(spark=spark, outputCol="popularity", proba_df_path=als_df_path)
        br = BernoulliResponse(seed=seed, inputCol='popularity', outputCol='response')
        self.model = PipelineModel(
            stages=[als_resp, br])

    def transform(self, df):
        return self.model.transform(df).drop("popularity")
