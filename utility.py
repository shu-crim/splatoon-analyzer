from enum import Enum


class EventKind(Enum):
    death = 1
    kill = 2

class Event:
    event_kind = None
    frame = -1.0
    def __init__(self, event_kind, frame):
        self.event_kind = event_kind
        self.frame = frame


class TimeManager:
    last_sec_anchor = -1
    frame_anchor = -1
    gametime_zero_frame = None #アンカー入力が無い場合、算出不能とする
    fps = 0
    def __init__(self, fps):
        self.fps = fps

    def setAnchor(self, last_sec, frame):
        self.last_sec_anchor = last_sec
        self.frame_anchor = frame
        self.gametime_zero_frame = int(self.frame_anchor - (300 - self.last_sec_anchor) * self.fps)

    def frameToGameTime(self, frame):
        return (frame - self.gametime_zero_frame) / self.fps if self.fps > 0 and self.gametime_zero_frame != None else -1.0
