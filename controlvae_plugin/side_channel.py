from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid


class CustomSideChannel(SideChannel):
    def __init__(self, shared_dict = None):
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        self.shared = shared_dict
        self.message = {}

    def on_message_received(self, msg: IncomingMessage) -> None:
        # Handle incoming messages here
        key = msg.read_string()
        data = msg.read_float32_list()
        print(f"\nReceived message: {key}: \n{data}\n")
        self.message[key] = data
        self.shared[key] = data


    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)
    