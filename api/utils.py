from datetime import datetime
import os
import time
import random

class Uid:
    @staticmethod
    def create():
        timestamp = int(time.time() * 1000000)  # microseconds
        random_part = random.randint(1000, 9999)
        return f"{timestamp}{random_part}"