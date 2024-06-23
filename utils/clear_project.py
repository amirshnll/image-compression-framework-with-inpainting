import os
import shutil
from tqdm import tqdm


class ClearProject:
    def run(self) -> None:
        # Change to the parent directory (main project directory)
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Clear the main project directory
        self.clear_directory(".")

        # Clear the specified data directories
        self.clear_data_directories()

    def clear_directory(self, path: str) -> None:
        """
        Clears unnecessary files and directories within a given path.
        Removes __pycache__, .DS_Store, and other unwanted files/directories.
        """
        for root, dirs, files in tqdm(os.walk(path, topdown=False), desc="Clear Cache"):
            for dir in dirs:
                if dir == "__pycache__":
                    try:
                        shutil.rmtree(os.path.join(root, dir))
                        print(
                            f"Removed __pycache__ directory: {os.path.join(root, dir)}"
                        )
                    except OSError as e:
                        print(f"Error removing __pycache__: {e}")

            for file in files:
                if file == ".DS_Store":
                    try:
                        os.remove(os.path.join(root, file))
                        print(f"Removed .DS_Store file: {os.path.join(root, file)}")
                    except OSError as e:
                        print(f"Error removing .DS_Store: {e}")

    def clear_data_directories(self) -> None:
        """
        Clears specific directories and their contents within the 'data' folder.
        Specifically, removes all files except __init__.py from 'channel',
        'compressed', 'inpainting', 'restored', 'mask', and 'destination' directories.
        """
        data_dirs = [
            "data/channel",
            "data/compressed",
            "data/inpainting",
            "data/restored",
            "data/mask",
            "data/destination",
            "data/highlight",
            "data/plot",
        ]
        for data_dir in data_dirs:
            data_dir_path = os.path.join(".", data_dir)
            if os.path.exists(data_dir_path):
                for file in os.listdir(data_dir_path):
                    file_path = os.path.join(data_dir_path, file)
                    if os.path.isfile(file_path) and file != "__init__.py":
                        try:
                            os.remove(file_path)
                            print(f"Removed file: {file_path}")
                        except OSError as e:
                            print(f"Error removing file: {e}")

    def clear_data_directory(self, data_dir: str) -> None:
        """
        Clears specific directories and their contents within the 'data' folder.
        Specifically, removes all files except __init__.py from the specified directory.
        """
        self.clear_directory(data_dir)

        data_dir_path = os.path.join(".", data_dir)
        if os.path.exists(data_dir_path):
            for file in os.listdir(data_dir_path):
                file_path = os.path.join(data_dir_path, file)
                if os.path.isfile(file_path) and file != "__init__.py":
                    try:
                        os.remove(file_path)
                        print(f"Removed file: {file_path}")
                    except OSError as e:
                        print(f"Error removing file: {e}")
