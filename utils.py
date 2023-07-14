import matplotlib.pyplot as plt
import pyspark.sql.functions as sf
from IPython.display import clear_output


def plot_metric(metrics):
    clear_output(wait=True)
    plt.plot(metrics)
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('# of clicks')
    plt.show()


def calc_metric(response_df):
    return (response_df
            .groupBy("user_idx").agg(sf.sum("response").alias("num_positive"))
            .select(sf.mean("num_positive")).collect()[0][0]
           )
