from .encoding import Encoding


class Channel:
    """
    This class represents a channel in an image processing system.
    It encapsulates two main functions: encode and transmit data using the specified
    encoding method, and receive and decode data transmitted through this channel.
    """

    def __init__(self) -> None:
        """
        Initializes the class instance.
        """
        self.encoding = Encoding()

    def encode_and_transmit(self, data: str, encoding_method: str) -> str:
        """
        Encodes the given data using the specified encoding method
        and transmits it through this channel.
        """
        # Encode the data using the specified encoding method
        encoded_data = self.encoding.encode(data, encoding_method)
        return encoded_data

    def receive_and_decode(self, transmitted_data: str, encoding_method: str) -> str:
        """
        Receives the given transmitted data through this channel and decodes it
        using the specified encoding method.
        """
        # Decode the transmitted data using the specified encoding method
        decoded_data = self.encoding.decode(transmitted_data, encoding_method)
        return decoded_data
