import numpy as np
import cv2
from enum import Enum


class IkaLampReader:
    roi_top_lamps = 41
    roi_bottom_lamps = 101
    roi_left_our_lamps = 500
    roi_left_opponent_lamps = 1050
    roi_width_lamps = 370
    cross_lumi = 104
    span_lamps = 64
    num_lumps_per_team = 4
    th_cross_match = 0.8

    def __init__(self, cross_template_img_path):
        self.img_cross_template = cv2.cvtColor(cv2.imread(cross_template_img_path), cv2.COLOR_BGR2GRAY)
        self.mask_cross = self.img_cross_template != 0
        self.cross_px = np.sum(self.mask_cross)

    def read(self, img_capture):
        our_lamp = self.readFourLamps(img_capture[self.roi_top_lamps:self.roi_bottom_lamps, self.roi_left_our_lamps:self.roi_left_our_lamps+self.roi_width_lamps])
        opponent_lamp = self.readFourLamps(img_capture[self.roi_top_lamps:self.roi_bottom_lamps, self.roi_left_opponent_lamps:self.roi_left_opponent_lamps+self.roi_width_lamps])
        return our_lamp, opponent_lamp
        
    def readFourLamps(self, roi_lamps):
        width_cross = self.mask_cross.shape[1]
        img_sub_cross_lumi = np.abs(roi_lamps.astype(int) - self.cross_lumi)
        img_sub_cross_lumi = np.sum(img_sub_cross_lumi, axis=2)

        match_width = roi_lamps.shape[1] - width_cross
        match = np.zeros((match_width))
        for x in range(match_width):
            roi = img_sub_cross_lumi[:, x:x+width_cross]
            match[x] = 1 - sum(roi[self.mask_cross]) / self.cross_px / 255

        num_cross = 0
        for i in range(self.num_lumps_per_team):
            # plt.plot(match)
            # plt.show()
            x_max = np.argmax(match)
            if (match[x_max] >= self.th_cross_match):
                num_cross += 1
                match[max(0, x_max-self.span_lamps//2):min(x_max+self.span_lamps//2, match.shape[0])] = 0
            else:
                break

        # 現段階では回線落ちを未考慮。しかも数のみ
        return self.num_lumps_per_team - num_cross


class TimeReader:
    img_digit = []

    def __init__(self, digit_template_paths):
        for i in range(10):
            self.img_digit.append(cv2.imread(digit_template_paths[i]))

    def read(self, img_capture):
        # 残り時間読み
        digit_width = 24
        th_digit_similarity = 0.8
        roi_digit = img_capture[54:94, 915:1005]
        roi_digit = cv2.cvtColor(roi_digit, cv2.COLOR_BGR2GRAY)
        roi_digit = cv2.cvtColor(roi_digit, cv2.COLOR_GRAY2BGR)

        # 3つの数字を読む
        result_sec_under = np.zeros((10, 1, 1), np.float32)
        result_sec_upper = np.zeros((10, 1, 1), np.float32)
        result_min = np.zeros((10, 1, 1), np.float32)
        roi_sec_under = roi_digit[:, 65:65+digit_width, :]
        roi_sec_upper = roi_digit[:, 41:41+digit_width, :]
        roi_min = roi_digit[:, 0:digit_width, :]
        for i in range(10):
            result_sec_under[i] = cv2.matchTemplate(roi_sec_under, self.img_digit[i], cv2.TM_CCOEFF_NORMED)
            result_sec_upper[i] = cv2.matchTemplate(roi_sec_upper, self.img_digit[i], cv2.TM_CCOEFF_NORMED)
            result_min[i] = cv2.matchTemplate(roi_min, self.img_digit[i], cv2.TM_CCOEFF_NORMED)

        read_sec_under = np.argmax(result_sec_under) if np.max(result_sec_under) >= th_digit_similarity else -1
        read_sec_upper = np.argmax(result_sec_upper) if np.max(result_sec_upper) >= th_digit_similarity else -1
        read_min = np.argmax(result_min) if np.max(result_min) >= th_digit_similarity else -1
        read_last_sec = read_min * 60 + read_sec_upper * 10 + read_sec_under if read_sec_under >= 0 and read_sec_upper >= 0 and read_min >= 0 else -1
        # print(str.format("last time[sec] {0:03d} {1:02d}:{2}{3} ({4:0.2f}:{5:0.2f} {6:0.2f})", read_last_sec, read_min, read_sec_upper, read_sec_under, np.max(result_min), np.max(result_sec_upper), np.max(result_sec_under)))

        # cv2.imshow('Video', roi_digit)
        # cv2.waitKey(1)

        return read_last_sec


class GachiKind(Enum):
    none = 0
    area = 1
    yagura = 2
    hoko = 3
    asari = 4

class CountReader:
    th_100match = 0.6
    th_digit_match = 0.6
    gachi_kind = GachiKind.area #GachiKind.none
    def __init__(self, path_template_count100):
        img = cv2.imread(path_template_count100)
        self.img_template_count100 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 数字のテンプレートを読み込む
        self.img_template_digit = []
        for i in range(10):
            self.img_template_digit.append(cv2.cvtColor(cv2.imread(r".\template\digit_" + str(i) + ".png"), cv2.COLOR_BGR2GRAY))
        self.digit_height, self.digit_width = self.img_template_digit[0].shape[:2]

    def read(self, img_capture):
        count = (-1, -1)
        if self.gachi_kind == GachiKind.none:
            # 種類の特定
            # ガチエリア：味方のカウント100を認識
            if self.readAreaCount100(img_capture):
                self.gachi_kind = GachiKind.area

        if self.gachi_kind == GachiKind.area:
            # ガチエリアのカウント読み
            count = self.readAreaCount(img_capture)

        return self.gachi_kind, count

    def readAreaCount100(self, img_capture):
        roi = img_capture[150:210, 810:890]
        img_bin = self.rotateAndBinarize(roi, 5)
        # cv2.imshow('Video', img_bin)
        # cv2.waitKey(1)
        match = cv2.matchTemplate(img_bin, self.img_template_count100, cv2.TM_CCOEFF_NORMED)
        # print("100match: " + str(match))
        if np.max(match) >= self.th_100match:
            return True
        else:
            return False

    def readAreaCount(self, img_capture):
        count_self = -1
        count_opponent = -1
        # 味方
        roi = img_capture[150:210, 810:890]
        img_bin = self.rotateAndBinarize(roi, 5)
        
        # 100とマッチング
        match = cv2.matchTemplate(img_bin, self.img_template_count100, cv2.TM_CCOEFF_NORMED)
        if np.max(match) >= self.th_100match:
            count_self = 100
        else:
            # 縮小して数字テンプレートのサイズに合わせる
            img_resize = cv2.resize(img_bin, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
            # cv2.imshow('count self', img_resize)

            # 数字読み
            count_self = self.readTwoDigits(img_resize)
        
        # 相手
        roi = img_capture[150:210, 1035:1115]
        img_bin = self.rotateAndBinarize(roi, -5)

        # 100とマッチング
        match = cv2.matchTemplate(img_bin, self.img_template_count100, cv2.TM_CCOEFF_NORMED)
        if np.max(match) >= self.th_100match:
            count_opponent = 100
        else:
            # 縮小して数字テンプレートのサイズに合わせる
            img_resize = cv2.resize(img_bin, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
            # cv2.imshow('count opponent', img_resize)

            # 数字読み
            count_opponent = self.readTwoDigits(img_resize)
        # cv2.waitKey(1)

        return (count_self, count_opponent)

    def rotateAndBinarize(self, img, degree):
        # グレースケール化
        roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 回転
        h, w = roi_gray.shape[:2]
        center = (int(w / 2), int(h / 2))
        trans = cv2.getRotationMatrix2D(center, degree, 1.0)
        img_rotate = cv2.warpAffine(roi_gray, trans, (w,h))
        
        # 数字領域ぎりぎりに絞ることでコントラストの低い数字と背景を分離できるようにする
        roi_binarize = img_rotate[5:-5, 5:-5]

        # 二値化
        ret, img_otsu = cv2.threshold(roi_binarize, 0, 255, cv2.THRESH_OTSU)
        img_binarize = np.zeros(img_rotate.shape, dtype=img_rotate.dtype)
        img_binarize[5:-5, 5:-5] = img_otsu

        return img_binarize

    def readTwoDigits(self, img_digits):
        # 2桁
        num_left = self.matchDigit(img_digits[4:4+self.digit_height, 7:7+self.digit_width])
        num_right = self.matchDigit(img_digits[4:4+self.digit_height, 30:30+self.digit_width])
        if num_left >= 0 and num_right >= 0:
            num = num_left * 10 + num_right
        else:
            # 1桁
            num = self.matchDigit(img_digits[4:4+self.digit_height, 18:18+self.digit_width])
        return num

    def matchDigit(self, img_digit):
        result = np.zeros((10, 1, 1), np.float32)
        for i in range(10):
            result[i] = cv2.matchTemplate(img_digit, self.img_template_digit[i], cv2.TM_CCOEFF_NORMED)
        return np.argmax(result) if np.max(result) >= self.th_digit_match else -1


class KillSearcher:
    def __init__(self, template_img_path, th_taoshita_match = 0.8):
        self.img_template = cv2.imread(template_img_path)
        self.th_taoshita_match = th_taoshita_match
    def find(self, img_capture):
        roi = img_capture[790:790+(4*65), 850:1180]

        match = cv2.matchTemplate(roi, self.img_template, cv2.TM_CCOEFF_NORMED)
        slot_max = np.zeros((4))
        for i in range(4):
            y = (3*65) - i * 65
            roi_i = match[y:y+25, :]
            slot_max[i] = np.max(roi_i)
        found = slot_max >= self.th_taoshita_match

        np.set_printoptions(precision=2)
        # print("kill match: ", slot_max)
        # cv2.imshow('Video', roi)
        # cv2.waitKey(0)
        return np.sum(found)


class DeathSearcher:
    roi_top = 350
    roi_bottom = 500
    roi_left = 850
    roi_right = 1050
    find_span_sec = 10

    def __init__(self, template_img_path, th_yarareta_match = 0.92, fps = 30):
        self.img_template = cv2.imread(template_img_path)
        self.th_yarareta_match = th_yarareta_match
        self.fps = fps
        self.found_frame = -self.find_span_sec * fps

    def find(self, img_capture, index_frame):
        if index_frame <= self.found_frame + self.find_span_sec * self.fps:
            return False

        roi = img_capture[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right] 
        result = cv2.matchTemplate(roi, self.img_template, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        # #print(f"max value: {maxVal}, position: {maxLoc}")

        if maxVal >= self.th_yarareta_match:
            self.found_frame = index_frame
            return True
        else:
            return False


