from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame

import pyspark.sql.functions as sf

from sim4rec.response import BernoulliResponse, ActionModelTransformer

from response_models.utils import get_session


class PopBasedDriftTransformer(ActionModelTransformer):

    def __init__(
            self,
            spark: SparkSession,
            outputCol: str = None,
            pop_df_path: str = None,
            drift_start: int = 5,
            drift_end: int = 15,
            drift_base: float = 0.06,
    ):
        """
        Decrease popularity of popular items by `drift_base` each iteration between `drift_start` and `drift_end`.

        :param outputCol: Name of the response probability column
        :param pop_df_path: path to a spark dataframe with items' popularity
        :param drift_start: iteration, when preference drift starts
        :param drift_end: iteration, when preference drift ends
        :param drift_base: score change on each preference drift iteration
        """
        self.pop_df = sf.broadcast(spark.read.parquet(pop_df_path))
        self.outputCol = outputCol
        self.drift_start = drift_start
        self.drift_end = drift_end
        self.drift_base = drift_base
        self.iteration = 1

    def set_iteration(self, iteration):
        self.iteration = iteration

    def get_popularity(self, iteration: int) -> DataFrame:
        num_drifts = min(max(iteration - self.drift_start, 0), self.drift_end - self.drift_start)
        return (self.pop_df.withColumn(self.outputCol,
                                       sf.when(sf.col("is_popular") == 1,
                                               sf.col(self.outputCol) - sf.lit(
                                                   self.drift_base * num_drifts))
                                       .otherwise(sf.when(sf.col("is_unpopular") == 1,
                                                          sf.col(self.outputCol) + sf.lit(
                                                              self.drift_base * num_drifts))
                                                  .otherwise(sf.col(self.outputCol))))
                .withColumn(self.outputCol, sf.when(sf.col(self.outputCol) < sf.lit(0.), sf.lit(0.))
                            .otherwise(sf.when(sf.col(self.outputCol) > sf.lit(1.), sf.lit(1.))
                                       .otherwise(sf.col(self.outputCol))))
                )

    def _transform(self, dataframe):
        current_response = self.get_popularity(self.iteration)
        self.iteration += 1
        return (dataframe
                .join(current_response, on='item_idx')
                .drop(*set(self.pop_df.columns).difference(["item_idx", self.outputCol]))
                )


class TaskThreeResponse:
    def __init__(self, spark, pop_df_path="./response_models/data/popular_items_popularity.parquet", seed=123):
        pop_resp = PopBasedDriftTransformer(spark=spark, outputCol="popularity", pop_df_path=pop_df_path)
        br = BernoulliResponse(seed=seed, inputCol='popularity', outputCol='response')
        self.model = PipelineModel(
            stages=[pop_resp, br])

    def transform(self, df):
        return self.model.transform(df).drop("popularity")


if __name__ == '__main__':
    import pandas as pd

    spark = get_session()
    task = TaskThreeResponse(spark, pop_df_path="./data/popular_items_popularity.parquet", seed=123)
    test_df = spark.createDataFrame(pd.DataFrame({"item_idx": [3, 5, 1], "user_idx": [5, 2, 1]}))
    print("iteration", task.model.stages[0].iteration)
    task.model.stages[0].get_popularity(1).show()
    task.transform(test_df).show()
    task.model.stages[0].iteration = 4
    print("iteration", task.model.stages[0].iteration)
    task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
    task.transform(test_df).show()
    print("iteration", task.model.stages[0].iteration)
    task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
    task.transform(test_df).show()
    print("iteration", task.model.stages[0].iteration)
    task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
    task.transform(test_df).show()
    task.model.stages[0].iteration = 14
    print("iteration", task.model.stages[0].iteration)
    task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
    task.transform(test_df).show()
    print("iteration", task.model.stages[0].iteration)
    task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
    task.transform(test_df).show()
    print("iteration", task.model.stages[0].iteration)
    task.model.stages[0].get_popularity(task.model.stages[0].iteration).show()
    task.transform(test_df).show()