from enum import IntEnum


class ActorStatus(IntEnum):
    READY = 0
    ACTIVATE = 1
    UP = 2
    HIT = 3
    DOWN = 4
    RESET = 5
