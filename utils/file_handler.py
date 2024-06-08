class FileHandler:
    """
    A class to handle basic file operations such as reading, writing, and appending text to a file.
    """

    def __init__(self, file_path: str):
        """
        Initializes the FileHandler with the path to the file.
        """
        self.file_path = file_path

    def read_file(self) -> str:
        """
        Reads the content of the file and returns it as a string.
        """
        try:
            with open(self.file_path, "r") as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return "File not found."

    def write_file(self, content: str) -> str:
        """
        Writes the given content to the file, overwriting any existing content.
        """
        try:
            with open(self.file_path, "w") as file:
                file.write(content)
            return "Write operation successful."
        except Exception as e:
            return f"An error occurred: {e}"

    def append_file(self, content: str) -> str:
        """
        Appends the given content to the file.
        """
        try:
            with open(self.file_path, "a") as file:
                file.write(content)
            return "Append operation successful."
        except Exception as e:
            return f"An error occurred: {e}"

    def write_binary_file(self, content: bytes) -> str:
        """
        Writes the given binary content to the file, overwriting any existing content.
        """
        try:
            with open(self.file_path, "wb") as file:
                file.write(content)
            return "Binary write operation successful."
        except Exception as e:
            return f"An error occurred: {e}"
