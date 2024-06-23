import os
import matplotlib.pyplot as plt


class Plot:
    def show_plot(
        self,
        data: list[dict],
        plot_type: str = "image",
        title: str | None = None,
        caption: str | None = None,
        subplot_layout: tuple[int, int] | None = None,
    ) -> None:
        """
        :param data: list of dictionaries with 'title' and 'image' or 'data' keys
        :param plot_type: 'image' for image data, 'plot' for regular data plots
        :param title: overall title for the entire plot
        :param caption: overall caption for the entire plot
        :param subplot_layout: tuple indicating the layout (rows, columns)
        """
        if subplot_layout:
            rows, cols = subplot_layout
        else:
            rows, cols = 1, len(data)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 8), squeeze=False)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=20, wspace=20)

        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                if plot_type == "image":
                    ax.imshow(data[i * cols + j]["image"], cmap="gray")
                else:
                    ax.plot(data[i * cols + j]["data"])
                ax.set_title(data[i * cols + j]["title"])
                ax.axis("off")

        if title:
            fig.suptitle(title, fontsize=16)
        if caption:
            plt.figtext(
                0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=12
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def save_plot(
        self,
        data: list[dict],
        plot_type: str = "image",
        title: str | None = None,
        filename_prefix: str = "plot",
        format: str = "png",
        subplot_layout: tuple[int, int] | None = None,
    ) -> None:
        """
        Saves the plot as image file(s) using matplotlib.
        """
        if subplot_layout:
            rows, cols = subplot_layout
        else:
            rows, cols = 1, len(data)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 8), squeeze=False)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=20, wspace=20)

        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                if plot_type == "image":
                    ax.imshow(data[i * cols + j]["image"], cmap="gray")
                ax.set_title(data[i * cols + j]["title"])
                ax.axis("off")

        if title:
            fig.suptitle(title, fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Create directory if it doesn't exist
        save_dir = "data/plot"
        os.makedirs(save_dir, exist_ok=True)

        # Save each subplot as a separate file
        for idx, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                # Properly format the filename using f-string
                filename = f"{filename_prefix}-{data[i * cols + j]['title']}-{idx}.{format}"
                filepath = os.path.join(save_dir, filename)
                fig.savefig(filepath, format=format)

        plt.close(fig)  # Close the figure to release memory

        print(f"Plots saved in {save_dir}")
