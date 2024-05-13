import matplotlib.pyplot as plt

from utils.constants import MATERIAL_COLORS, SEASON_COLORS, logger


def plot_histogram(data, season_hist, title, x_label, y_label, legend_title, file_name):
    try:
        sorted_keys = sorted(data.keys())
        time_points = range(len(season_hist))

        fig, ax = plt.subplots(figsize=(10, 6))

        lines = []
        for key in sorted_keys:

            if key not in MATERIAL_COLORS:
                color = plt.cm.tab20(sorted_keys.index(key) % 20)
            else:
                color = MATERIAL_COLORS.get(key, "black")
            (line,) = ax.plot(time_points, data[key], label=key, color=color)
            lines.append(line)

        current_season_base = season_hist[0].split()[0]
        start_index = 0
        for i, season in enumerate(season_hist):
            base_season = season.split()[0]  # Extract base season name
            if base_season != current_season_base or i == len(season_hist) - 1:
                end_index = i if i < len(season_hist) - 1 else i + 1
                ax.axvspan(
                    start_index,
                    end_index,
                    color=SEASON_COLORS.get(current_season_base, "grey"),
                    alpha=0.2,
                )
                current_season_base = base_season
                start_index = i

        data_legend = ax.legend(
            lines,
            [l.get_label() for l in lines],
            loc="upper left",
            title=legend_title,
            bbox_to_anchor=(1.05, 1),
        )
        ax.add_artist(data_legend)

        season_patches = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=color,
                markersize=10,
            )
            for season, color in SEASON_COLORS.items()
        ]
        season_legend = ax.legend(
            season_patches,
            SEASON_COLORS.keys(),
            title="Seasons",
            loc="upper left",
            bbox_to_anchor=(1.05, 0.6),
        )

        ax.add_artist(season_legend)

        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.subplots_adjust(right=0.7)
        plt.savefig(file_name)
        logger.info(f"Saved plot to {file_name}")
    except Exception as e:
        logger.error(f"Error plotting histogram: {e}")


def filter_and_plot_histogram(
    hist_data,
    season_hist,
    title,
    x_label,
    y_label,
    legend_title,
    file_name,
    filter_keys=None,
):
    if filter_keys is None:
        filter_keys = set(hist_data[0].keys())

    data = {key: [] for key in filter_keys}
    for counter in hist_data:
        for key in data:
            data[key].append(counter.get(key, 0))

    plot_histogram(data, season_hist, title, x_label, y_label, legend_title, file_name)


