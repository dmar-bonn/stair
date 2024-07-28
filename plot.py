import click
import seaborn as sb
import matplotlib.pyplot as plt
import pandas
import os
import json
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

PLOT_FONT_SIZE = 35
PLOT_LEGEND_SIZE = 25
PLOT_TICKS_SIZE = 35
PLOT_LINE_WIDTH = 10
IMG_FORMAT = "png"

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams["figure.autolayout"] = False
sb.set_palette("bright")


@click.command()
@click.option(
    "--scene_path", required=True, type=str, help="path to experiment on test scene"
)
def main(scene_path):
    total_df = pandas.DataFrame(
        {
            "Planner Type": [],
            "Step": [],
            "PSNR": [],
            "SSIM": [],
            "Depth Error": [],
            "Uncertainty": [],
            "Chamfer Distance": [],
            "Recall": [],
            "Precision": [],
            "F1-Score": [],
        }
    )

    planner_list = [
        c for c in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, c))
    ]
    # planner_list = [
    #     "Ours",
    #     "Exploration",
    #     "Fixed Pattern",
    #     "Max. View Distance",
    #     "Uniform",
    # ]
    # planner_list = [
    #     "Ours",
    #     "Ours(Explicit)",
    #     "STE",
    #     "STE(Implicit)",
    # ]
    # planner_list = [
    #     "ε=0.2",
    #     "ε=0.5",
    #     "ε=0.8",
    #     "ε=0.0",
    # ]
    print(planner_list)
    for planner in planner_list:
        planner_path = f"{scene_path}/{planner}"
        id_list = [
            int(c)
            for c in os.listdir(planner_path)
            if os.path.isdir(os.path.join(planner_path, c))
        ]
        print(id_list)
        for i, id in enumerate(id_list):
            data_path = f"{planner_path}/{id}/results.json"
            with open(data_path, "r") as json_file:
                result_data = json.load(json_file)

            for step, frame in result_data.items():
                dataframe = pandas.DataFrame(
                    {
                        "Planner Type": planner,
                        "Step": step,
                        "PSNR": (
                            frame["average_psnr"]
                            if "average_psnr" in frame.keys()
                            else 0
                        ),
                        "SSIM": (
                            frame["average_ssim"]
                            if "average_ssim" in frame.keys()
                            else 0
                        ),
                        "Depth Error": (
                            frame["average_depth_error"]
                            if "average_depth_error" in frame.keys()
                            else 0
                        ),
                        "Uncertainty": (
                            frame["average_uncertainty"]
                            if "average_uncertainty" in frame.keys()
                            else 0
                        ),
                        "Chamfer Distance": (
                            frame["chamfer_distance"]
                            if "chamfer_distance" in frame.keys()
                            else 0
                        ),
                        "Recall": frame["recall"] if "recall" in frame.keys() else 0,
                        "Precision": (
                            frame["precision"] if "precision" in frame.keys() else 0
                        ),
                        "F1-Score": frame["f1"] if "f1" in frame.keys() else 0,
                    },
                    index=[i],
                )
                total_df = total_df.append(dataframe)

    fig, ax = plt.subplots()
    plot_ax(ax, "Uncertainty", total_df)
    plt.savefig(f"{scene_path}/uncertainty.{IMG_FORMAT}", bbox_inches="tight")
    plt.clf()

    fig, ax = plt.subplots()
    plot_ax(ax, "PSNR", total_df)
    plt.savefig(f"{scene_path}/psnr.{IMG_FORMAT}", bbox_inches="tight")
    plt.clf()

    fig, ax = plt.subplots()
    plot_ax(ax, "SSIM", total_df)
    plt.savefig(f"{scene_path}/ssim.{IMG_FORMAT}", bbox_inches="tight")
    plt.clf()

    fig, ax = plt.subplots()
    plot_ax(ax, "Depth Error", total_df)
    plt.savefig(f"{scene_path}/depth_error.{IMG_FORMAT}", bbox_inches="tight")
    plt.clf()

    fig, ax = plt.subplots()
    plot_ax(ax, "Chamfer Distance", total_df)
    plt.savefig(f"{scene_path}/chamfer_distance.{IMG_FORMAT}", bbox_inches="tight")
    plt.clf()

    fig, ax = plt.subplots()
    plot_ax(ax, "Recall", total_df)
    plt.savefig(f"{scene_path}/recall.{IMG_FORMAT}", bbox_inches="tight")
    plt.clf()

    fig, ax = plt.subplots()
    plot_ax(ax, "Precision", total_df)
    plt.savefig(f"{scene_path}/precision.{IMG_FORMAT}", bbox_inches="tight")
    plt.clf()

    fig, ax = plt.subplots()
    plot_ax(ax, "F1-Score", total_df)
    plt.savefig(f"{scene_path}/f1_score.{IMG_FORMAT}", bbox_inches="tight")

    plt.rcParams["figure.figsize"] = [5, 15]
    label_instances, label_names = ax.get_legend_handles_labels()
    for label in label_instances:
        label.set_linewidth(7)
    figl, axl = plt.subplots()
    axl.axis("off")
    axl.legend(
        label_instances,
        label_names,
        loc="center",
        ncol=1,
        fontsize=24,
        frameon=False,
        handlelength=3,
        labelspacing=0.1,
    )
    figl.savefig(f"{scene_path}/legend.{IMG_FORMAT}")
    plt.clf()


def plot_ax(ax, metric, dataframe):

    sb.lineplot(
        dataframe,
        x="Step",
        y=metric,
        hue="Planner Type",
        style="Planner Type",
        linewidth=PLOT_LINE_WIDTH,
        ax=ax,
        errorbar=("sd", 1),
        palette=["C2", "C0", "C3", "C7", "C5"],
        dashes=["", "", "", ""],
    )

    ax.set_ylabel(metric, fontsize=PLOT_FONT_SIZE)
    ax.set_xlabel("Planning Step", fontsize=PLOT_FONT_SIZE)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.tick_params(axis="both", labelsize=PLOT_TICKS_SIZE)
    ax.get_legend().remove()


if __name__ == "__main__":
    main()
