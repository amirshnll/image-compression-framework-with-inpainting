import graphviz
from typing import Dict, List, Tuple

# Instructions for Installing Graphviz on macOS and Windows

# Installing Graphviz on macOS:
# 1. Using Homebrew:
#    Homebrew is a popular package manager for macOS. If you don't have Homebrew installed,
#    you can install it by running the following command in your Terminal:
#    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#
# 2. Install Graphviz:
#    Once Homebrew is installed, you can install Graphviz by running:
#    brew install graphviz
#
# 3. Verify the Installation:
#    After installation, you can verify that Graphviz is installed correctly by running:
#    dot -V
#    This should print the version of Graphviz.

# Installing Graphviz on Windows:
# 1. Download the Installer:
#    Go to the Graphviz download page: https://graphviz.gitlab.io/download/
#    and download the appropriate installer for your version of Windows.
#
# 2. Run the Installer:
#    Execute the downloaded installer and follow the installation instructions.
#    Make sure to check the option to add Graphviz to your system PATH during the installation process.
#
# 3. Verify the Installation:
#    After installation, open a Command Prompt and verify the installation by running:
#    dot -V
#    This should print the version of Graphviz.

# Installing the Graphviz Python Package:
# After installing Graphviz on your system, you need to install the Graphviz Python package to use it in your Python scripts.
#
# 1. Install the graphviz Python package:
#    You can install it using pip. Run the following command:
#    pip install graphviz
#
# 2. Verify the Installation:
#    You can verify the installation by running a simple Python script to import graphviz and create a basic graph:
#
#    import graphviz
#    dot = graphviz.Digraph(comment="Test Graph")
#    dot.node("A", "Start")
#    dot.edge("A", "B")
#    print(dot.source)
#
#    If this script runs without any errors, then Graphviz and the graphviz Python package are installed correctly on your system.


class Flowchart:
    def __init__(self, title: str) -> None:
        self.dot = graphviz.Digraph(comment=title)
        self.title = title

    def add_nodes(self, nodes: Dict[str, str]) -> None:
        """
        Adds nodes to the flowchart, including those within subgraphs.
        """
        for node_id, node_label in nodes.items():
            self.dot.node(node_id, node_label)

    def add_edges(self, edges: List[Tuple[str, str]]) -> None:
        """
        Adds edges to the flowchart.
        """
        for edge in edges:
            self.dot.edge(edge[0], edge[1])

    def add_subgraphs(self, subgraphs: Dict[str, Dict[str, str]]) -> None:
        """
        Adds edges to the flowchart.
        """
        for subgraph_name, subgraph_data in subgraphs.items():
            with self.dot.subgraph(name=subgraph_name) as c:
                c.attr(label=subgraph_data["label"])
                for node_id, node_label in subgraph_data["nodes"].items():
                    c.node(node_id, node_label)

    def render(self, filename: str = "flowchart") -> None:
        """
        Renders the flowchart to a file in PNG format.
        """
        output_path = f"data/flowchart/{filename}"
        self.dot.render(output_path, format="png")
