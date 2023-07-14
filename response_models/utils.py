from pyspark.sql import SparkSession, DataFrame
from sim4rec.response import BernoulliResponse, ActionModelTransformer


def get_session(num_threads=4) -> SparkSession:
    return SparkSession.builder \
        .appName('simulator') \
        .master(f'local[{num_threads}]') \
        .config('spark.sql.shuffle.partitions', f'{num_threads * 3}') \
        .config('spark.default.parallelism', f'{num_threads * 3}') \
        .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC') \
        .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC') \
        .getOrCreate()


class ResponseTransformer(ActionModelTransformer):

    def __init__(
            self,
            spark: SparkSession,
            outputCol: str = None,
            proba_df_path: str = None,
    ):
        """
        Calculates users' response based on precomputed probability of item interaction

        :param outputCol: Name of the response probability column
        :param proba_df_path: path to a spark dataframe with precomputed user-item probability of interaction
        """
        self.proba_df = spark.read.parquet(proba_df_path)
        self.outputCol = outputCol

    def _transform(self, dataframe):
        return (dataframe
                .join(self.proba_df, on=['item_idx', 'user_idx'])
                .drop(*set(self.proba_df.columns).difference(["user_idx", "item_idx", self.outputCol]))
                )
