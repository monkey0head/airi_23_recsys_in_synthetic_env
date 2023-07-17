from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame

import pyspark.sql.functions as sf

from sim4rec.response import BernoulliResponse, ActionModelTransformer

# from utils import get_session
from pyspark.sql import SparkSession


def get_session(num_threads=4) -> SparkSession:
    return SparkSession.builder \
        .appName('simulator') \
        .master(f'local[{num_threads}]') \
        .config('spark.sql.shuffle.partitions', f'{num_threads * 3}') \
        .config('spark.default.parallelism', f'{num_threads * 3}') \
        .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC') \
        .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC') \
        .getOrCreate()


class BoosTransformer(ActionModelTransformer):

    def __init__(
            self,
            spark: SparkSession,
            outputCol: str = None,
            boost_df_path: str = None,
    ):
        """
        Response function based on gradient boosting trained on ML-1M

        :param outputCol: Name of the response probability column
        :param boost_df_path: path to a spark dataframe with items' popularity
        """
        self.boost_df = spark.read.parquet(boost_df_path)
        self.outputCol = outputCol

    def _transform(self, dataframe):
        return (dataframe
                .join(self.boost_df, on=['item_idx', 'user_idx'])
                .drop(*set(self.boost_df.columns).difference(["user_idx", "item_idx", self.outputCol]))
                )


class TaskTwoResponse:
    def __init__(self, spark, boost_df_path, seed=123):
        pop_resp = BoosTransformer(spark=spark, outputCol="popularity", boost_df_path=boost_df_path)
        br = BernoulliResponse(seed=seed, inputCol='popularity', outputCol='response')
        self.model = PipelineModel(
            stages=[pop_resp, br])

    def transform(self, df):
        return self.model.transform(df).drop("popularity")


# if __name__ == '__main__':
#     import pandas as pd

#     spark = get_session()
#     task = TaskTwoResponse(spark, pop_df_path="./data/popular_items_popularity.parquet", seed=123)
#     test_df = spark.createDataFrame(pd.DataFrame({"item_id": [3, 5, 1], "user_id": [5, 2, 1]}))
#     print("iteration", task.model.stages[0].iteration)
#     task.model.stages[0].get_popularity(1).show()
#     task.transform(test_df).show()
#     task.model.stages[0].iteration = 4
#     print("iteration", task.model.stages[0].iteration)
#     task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
#     task.transform(test_df).show()
#     print("iteration", task.model.stages[0].iteration)
#     task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
#     task.transform(test_df).show()
#     print("iteration", task.model.stages[0].iteration)
#     task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
#     task.transform(test_df).show()
#     task.model.stages[0].iteration = 14
#     print("iteration", task.model.stages[0].iteration)
#     task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
#     task.transform(test_df).show()
#     print("iteration", task.model.stages[0].iteration)
#     task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
#     task.transform(test_df).show()
#     print("iteration", task.model.stages[0].iteration)
#     task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
#     task.transform(test_df).show()
