import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from metric_utils import RunningTrainMetrics

sns.set_style("darkgrid")


def training_mae_numexamples_lineplot(running_metrics: RunningTrainMetrics) -> Figure:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(
        x=running_metrics.train_examples_num, y=running_metrics.train_maes, ax=ax
    )
    plt.xlabel("Total number of training examples seen across epochs")
    plt.ylabel("Mean absolute error over minibatch")
    return fig


def epoch_metrics_lineplot(epochs_metrics_df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(
        data=epochs_metrics_df,
        x="epoch_num",
        y="test_mae",
        label="test_mae",
        ax=ax,
    )
    sns.lineplot(
        data=epochs_metrics_df,
        x="epoch_num",
        y="train_mae",
        label="train_mae",
        ax=ax,
    )
    plt.legend()
    return fig
