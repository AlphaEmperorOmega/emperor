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


class TerminalZAxisOffsetOptions(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class TerminalConnectionShapeOptions(Enum):
    BOX = "box"
    CROSS = "cross"
    SPHERE = "sphere"
    DIAGONAL_X = "diagonal_x"
    LINE_LEFT_RIGHT = "line_left_right"
    LINE_UP_DOWN = "line_up_down"
    LINE_FRONT_BACK = "line_front_back"
