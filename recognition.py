import numpy as np
import cv2
from enum import Enum
import matplotlib.pyplot as plt


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
    th_offline_match = 0.8

    def __init__(self, cross_template_img_path, offline_template_img_path):
        self.img_cross_template = cv2.cvtColor(cv2.imread(cross_template_img_path), cv2.COLOR_BGR2GRAY)
        self.mask_cross = self.img_cross_template != 0
        self.cross_px = np.sum(self.mask_cross)
        self.img_offline_template = cv2.imread(offline_template_img_path)
        self.height_offline_template = self.img_offline_template.shape[0]

    def read(self, img_capture):
        our_lamp = self.readFourLamps(img_capture[self.roi_top_lamps:self.roi_bottom_lamps, self.roi_left_our_lamps:self.roi_left_our_lamps+self.roi_width_lamps])
        opponent_lamp = self.readFourLamps(img_capture[self.roi_top_lamps:self.roi_bottom_lamps, self.roi_left_opponent_lamps:self.roi_left_opponent_lamps+self.roi_width_lamps], True)
        return our_lamp, opponent_lamp
        
    def readFourLamps(self, roi_lamps, opponent=False):
        width_cross = self.mask_cross.shape[1]
        img_sub_cross_lumi = np.abs(roi_lamps.astype(int) - self.cross_lumi)
        img_sub_cross_lumi = np.sum(img_sub_cross_lumi, axis=2)

        match_width = roi_lamps.shape[1] - width_cross
        match = np.zeros((match_width))
        for x in range(match_width):
            roi = img_sub_cross_lumi[:, x:x+width_cross]
            match[x] = 1 - sum(roi[self.mask_cross]) / self.cross_px / 255

        num_cross = 0
        lamps = [True, True, True, True]
        for i in range(self.num_lumps_per_team):
            # plt.plot(match)
            # plt.show()
            x_max = np.argmax(match)
            if (match[x_max] >= self.th_cross_match):
                num_cross += 1
                lamps[self.IconSenterXtoIndex(x_max, opponent)] = False
                match[max(0, x_max-self.span_lamps//2):min(x_max+self.span_lamps//2, match.shape[0])] = 0
            else:
                break

        # Search offline lamps
        num_offline = 0
        top_offline = (roi_lamps.shape[0] - self.height_offline_template) // 2
        roi_offline = roi_lamps[top_offline:top_offline+self.height_offline_template, :]
        offline_match = cv2.matchTemplate(roi_offline, self.img_offline_template, cv2.TM_CCOEFF_NORMED)[0]
        for i in range(self.num_lumps_per_team):
            x_max = np.argmax(offline_match)
            if (offline_match[x_max] >= self.th_offline_match):
                num_offline += 1
                lamps[self.IconSenterXtoIndex(x_max, opponent)] = False
                offline_match[max(0, x_max-self.span_lamps//2):min(x_max+self.span_lamps//2, match.shape[0])] = 0
            else:
                break

        # cv2.imshow('Lamp', roi_offline)
        # cv2.waitKey(0)

        return lamps

    def IconSenterXtoIndex(self, x, opponent=False):
        index = -1
        if not opponent:
            x = self.roi_width_lamps - self.mask_cross.shape[1] - x

        if x < 34:
            index = 0
        elif x < 134:
            index = 1
        elif x < 214:
            index = 2
        else:
            index = 3

        return index if opponent else 3 - index


class TimeReader:
    img_digit = []
    roi = (54, 94, 915, 1005)
    th_digit_similarity = 0.8
    left_upper_sec = 41
    left_lower_sec = 65
    th_extra_time_similarity = 0.7

    def __init__(self, digit_template_paths, extra_time_template_path):
        for i in range(10):
            self.img_digit.append(cv2.imread(digit_template_paths[i]))
        self.digit_width = self.img_digit[0].shape[1]
        self.img_extra_time_template = cv2.imread(extra_time_template_path)

    def read(self, img_capture):
        self.digit_width = 24
        roi_digit = img_capture[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        # 延長中認識
        match = cv2.matchTemplate(roi_digit, self.img_extra_time_template, cv2.TM_CCOEFF_NORMED)
        if match[0, 0] >= self.th_extra_time_similarity:
            return -1

        # 残り時間読み
        roi_digit = cv2.cvtColor(roi_digit, cv2.COLOR_BGR2GRAY)
        roi_digit = cv2.cvtColor(roi_digit, cv2.COLOR_GRAY2BGR)

        # 3つの数字を読む
        result_min = np.zeros((10, 1, 1), np.float32)
        result_sec_upper = np.zeros((10, 1, 1), np.float32)
        result_sec_lower = np.zeros((10, 1, 1), np.float32)
        roi_min = roi_digit[:, :self.digit_width, :]
        roi_sec_upper = roi_digit[:, self.left_upper_sec:self.left_upper_sec+self.digit_width, :]
        roi_sec_lower = roi_digit[:, self.left_lower_sec:self.left_lower_sec+self.digit_width, :]
        for i in range(10):
            result_min[i] = cv2.matchTemplate(roi_min, self.img_digit[i], cv2.TM_CCOEFF_NORMED)
            result_sec_upper[i] = cv2.matchTemplate(roi_sec_upper, self.img_digit[i], cv2.TM_CCOEFF_NORMED)
            result_sec_lower[i] = cv2.matchTemplate(roi_sec_lower, self.img_digit[i], cv2.TM_CCOEFF_NORMED)

        read_sec_lower = np.argmax(result_sec_lower) if np.max(result_sec_lower) >= self.th_digit_similarity else -1
        read_sec_upper = np.argmax(result_sec_upper) if np.max(result_sec_upper) >= self.th_digit_similarity else -1
        read_min = np.argmax(result_min) if np.max(result_min) >= self.th_digit_similarity else -1
        read_last_sec = read_min * 60 + read_sec_upper * 10 + read_sec_lower if read_sec_lower >= 0 and read_sec_upper >= 0 and read_min >= 0 else None
        # print(str.format("last time[sec] {0:03d} {1:02d}:{2}{3} ({4:0.2f}:{5:0.2f} {6:0.2f})", read_last_sec, read_min, read_sec_upper, read_sec_lower, np.max(result_min), np.max(result_sec_upper), np.max(result_sec_lower)))

        # cv2.imshow('Time', roi_digit)
        # cv2.waitKey(0)

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
    gachi_kind = GachiKind.area #GachiKind.none #試合の途中からでもルール認識できるようになるまでデバッグでarea固定
    roi_penalty_top = 222
    roi_penalty_bottom = 256
    roi_our_penalty_left = 863
    roi_opponent_penalty_left = 1038
    roi_penalty_width = 40
    num_buffer_frame_penalty = 5
    
    def __init__(self, path_template_count100):
        img = cv2.imread(path_template_count100)
        self.img_template_count100 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 数字のテンプレートを読み込む
        self.img_template_digit = []
        for i in range(10):
            self.img_template_digit.append(cv2.cvtColor(cv2.imread(r".\template\digit_" + str(i) + ".png"), cv2.COLOR_BGR2GRAY))
        self.digit_height, self.digit_width = self.img_template_digit[0].shape[:2]

        # ペナルティを読んだ結果のバッファ
        self.our_pnealty_buffer = [-1 for x in range(self.num_buffer_frame_penalty)]
        self.opponent_pnealty_buffer = [-1 for x in range(self.num_buffer_frame_penalty)]

    def read(self, img_capture):
        count = (-1, -1)
        penalty = (0, 0)
        if self.gachi_kind == GachiKind.none:
            # 種類の特定
            # ガチエリア：味方のカウント100を認識
            if self.readAreaCount100(img_capture):
                self.gachi_kind = GachiKind.area

        if self.gachi_kind == GachiKind.area:
            # ガチエリアのカウント読み
            count = self.readAreaCount(img_capture)
            # ペナルティ読み
            penalty = self.readAreaPenalty(img_capture)

        return self.gachi_kind, count, penalty

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
            count_self = self.readTwoDigitsOld(img_resize)
        
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
            count_opponent = self.readTwoDigitsOld(img_resize)
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

    def readTwoDigitsOld(self, img_digits):
        # 2桁
        num_left = self.matchDigit(img_digits[4:4+self.digit_height, 7:7+self.digit_width])
        num_right = self.matchDigit(img_digits[4:4+self.digit_height, 30:30+self.digit_width])
        if num_left >= 0 and num_right >= 0:
            num = num_left * 10 + num_right
        else:
            # 1桁
            num = self.matchDigit(img_digits[4:4+self.digit_height, 18:18+self.digit_width])
        return num

    def matchDigit(self, img_digit, th_digit_match = None):
        th_digit_match = self.th_digit_match if th_digit_match == None else th_digit_match
        result = np.zeros((10, 1, 1), np.float32)
        for i in range(10):
            result[i] = cv2.matchTemplate(img_digit, self.img_template_digit[i], cv2.TM_CCOEFF_NORMED)
        # print(str.format("digit:{0} match:{1}", np.argmax(result), np.max(result)))
        return np.argmax(result) if np.max(result) >= th_digit_match else -1

    def readAreaPenalty(self, img_capture):
        # 味方 
        roi_our_penalty = img_capture[self.roi_penalty_top:self.roi_penalty_bottom, self.roi_our_penalty_left:self.roi_our_penalty_left+self.roi_penalty_width]
        gray_our_penalty = cv2.cvtColor(roi_our_penalty, cv2.COLOR_BGR2GRAY)
        img_our_penalty_resized = cv2.resize(gray_our_penalty, dsize=(self.digit_width*2, self.digit_height), interpolation=cv2.INTER_NEAREST)
        ret, img_otsu = cv2.threshold(img_our_penalty_resized, 0, 255, cv2.THRESH_OTSU)
        our_pnealty = self.readTwoDigits(img_otsu, 0.65)
        # our_pnealty = 0 if our_pnealty < 0 else our_pnealty

        # バッファを詰め変える
        self.our_pnealty_buffer.append(our_pnealty)
        self.our_pnealty_buffer.pop(0)

        # バッファ内のペナルティカウントがすべて同じなら採択
        same = True
        for item in self.our_pnealty_buffer:
            if item != our_pnealty:
                same = False
                break
        if not same:
            our_pnealty = -1

        # print(self.our_pnealty_buffer)
        # print("our_pnealty:" + str(our_pnealty))
        # cv2.imshow('OurPenalty', img_otsu)
        # cv2.waitKey(1)
        
        # 敵
        roi_opponent_penalty = img_capture[self.roi_penalty_top:self.roi_penalty_bottom, self.roi_opponent_penalty_left:self.roi_opponent_penalty_left+self.roi_penalty_width]
        gray_opponent_penalty = cv2.cvtColor(roi_opponent_penalty, cv2.COLOR_BGR2GRAY)
        img_opponent_penalty_resized = cv2.resize(gray_opponent_penalty, dsize=(self.digit_width*2, self.digit_height), interpolation=cv2.INTER_NEAREST)
        opponent_pnealty = self.readTwoDigits(img_opponent_penalty_resized)

        # バッファを詰め変える
        self.opponent_pnealty_buffer.append(opponent_pnealty)
        self.opponent_pnealty_buffer.pop(0)

        # バッファ内のペナルティカウントがすべて同じなら採択
        same = True
        for item in self.opponent_pnealty_buffer:
            if item != opponent_pnealty:
                same = False
                break
        if not same:
            opponent_pnealty = -1

        return our_pnealty, opponent_pnealty

    def readTwoDigits(self, img_digits, th_digit_match = None): #後にこちらへ統一したい
        th_digit_match = self.th_digit_match if th_digit_match == None else th_digit_match

        # 画像サイズのチェック
        if img_digits.shape[0] != self.digit_height or img_digits.shape[1] != self.digit_width * 2:
            return -1

        # ROIの端に白画素がある場合、誤読リスクがあるので読まない
        if np.sum(img_digits[:1, :]) + np.sum(img_digits[-1:, :]) + np.sum(img_digits[:, :1]) + np.sum(img_digits[:, -1:]) > 0:
            return -1 

        # 2桁
        num_left = self.matchDigit(img_digits[:, :self.digit_width], th_digit_match)
        num_right = self.matchDigit(img_digits[:, self.digit_width:], th_digit_match)
        if num_left >= 0 and num_right >= 0:
            num = num_left * 10 + num_right
        else:
            # 1桁
            num = self.matchDigit(img_digits[:, self.digit_width//2:self.digit_width//2+self.digit_width], th_digit_match)
        return num


class KillSearcher:
    roi_left = 850
    roi_right = 1180
    roi_top = 790
    log_y_span = 65
    roi_height = 25
    num_max_log = 4

    def __init__(self, template_img_path, th_taoshita_match = 0.8):
        self.img_template = cv2.imread(template_img_path)
        self.th_taoshita_match = th_taoshita_match

    def find(self, img_capture):
        roi = img_capture[self.roi_top:self.roi_top+(self.num_max_log * self.log_y_span), self.roi_left:self.roi_right]

        match = cv2.matchTemplate(roi, self.img_template, cv2.TM_CCOEFF_NORMED)
        slot_max = np.zeros((self.num_max_log))
        for i in range(self.num_max_log):
            y = ((self.num_max_log - 1) * self.log_y_span) - i * self.log_y_span
            roi_i = match[y:y+self.roi_height, :]
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

    def __init__(self, template_img_path, fps = 30, th_yarareta_match = 0.92):
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

