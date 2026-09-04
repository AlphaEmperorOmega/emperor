from enum import Enum


class TerminalRangeOptions(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8


class TerminalConnectionShapeOptions(Enum):
    BOX = "box"
    CROSS = "cross"
    SPHERE = "sphere"
    DIAGONAL = "diagonal"
    CROSS_DIAGONAL = "cross_diagonal"


class TerminalRoutingTreeDepthOptions(Enum):
    TWO = 2
    THREE = 3
