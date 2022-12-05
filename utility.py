from enum import Enum
import cv2

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


class RecognitionMovieMaker:
    def __init__(self, path, fps, size):
        format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        self.video_writer = cv2.VideoWriter(path, format, fps, size)

    def makeFrame(self, img_src, last_sec, our_lamp, opponent_lamp, count, penalty, num_kill, death):
        # 残り時間
        if last_sec >= 0:
            cv2.putText(img_src,
                text="{0}:{1:02d}".format(last_sec // 60, last_sec % 60), org=(920, 140),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
        
        # ランプ
        text_lamp = [("O" if i == 1 else ("X" if i == 0 else "-")) for i in our_lamp]
        cv2.putText(img_src,
            text=str(text_lamp), org=(580, 130),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
        text_lamp = [("O" if i == 1 else ("X" if i == 0 else "-")) for i in opponent_lamp]
        cv2.putText(img_src,
            text=str(text_lamp), org=(1070, 130),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)

        # カウント
        cv2.putText(img_src,
            text=str(count[0]), org=(740, 200),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
        cv2.putText(img_src,
            text=str(count[1]), org=(1140, 200),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)

        # ペナルティ
        cv2.putText(img_src,
            text=str(penalty[0]), org=(740, 250),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
        cv2.putText(img_src,
            text=str(penalty[1]), org=(1140, 250),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)

        # キル
        cv2.putText(img_src,
            text=str("Kill:{0}".format(num_kill)), org=(1300, 1000),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)

        # デス
        if death:
            cv2.putText(img_src,
                text=str("Death".format(num_kill)), org=(840, 600),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)

        self.video_writer.write(img_src)

        cv2.imshow('output', img_src)
        cv2.waitKey(1)

    def release(self):
        self.video_writer.release()

