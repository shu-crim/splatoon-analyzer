import numpy as np
import cv2
from enum import Enum
import matplotlib.pyplot as plt
import time


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
    th_offline_match = 0.996

    def __init__(self, cross_template_img_path, offline_template_img_path):
        self.img_cross_template = cv2.cvtColor(cv2.imread(cross_template_img_path), cv2.COLOR_BGR2GRAY)
        self.mask_cross = self.img_cross_template != 0
        self.cross_px = np.sum(self.mask_cross)
        self.img_offline_template = cv2.imread(offline_template_img_path)
        self.height_offline_template, self.width_offline_template = self.img_offline_template.shape[:2]

    def read(self, img_capture):
        our_lamp = self.readFourLamps(img_capture[self.roi_top_lamps:self.roi_bottom_lamps, self.roi_left_our_lamps:self.roi_left_our_lamps+self.roi_width_lamps])
        opponent_lamp = self.readFourLamps(img_capture[self.roi_top_lamps:self.roi_bottom_lamps, self.roi_left_opponent_lamps:self.roi_left_opponent_lamps+self.roi_width_lamps], True)
        return our_lamp, opponent_lamp
        
    def readFourLamps(self, roi_lamps, opponent=False):
        width_cross = self.mask_cross.shape[1]
        img_sub_cross_lumi = np.abs(roi_lamps.astype(int) - self.cross_lumi)
        img_sub_cross_lumi = np.sum(img_sub_cross_lumi, axis=2)

        # 高速化のため画像を縮小して処理
        shrink_rate = 4
        img_sub_cross_lumi = img_sub_cross_lumi[::shrink_rate, ::shrink_rate]
        width_cross //= shrink_rate
        mask_cross = self.mask_cross[::shrink_rate, ::shrink_rate]
        cross_px = np.sum(mask_cross)
        span_lamps = self.span_lamps // shrink_rate

        time_start = time.perf_counter()
        match_width = img_sub_cross_lumi.shape[1] - width_cross
        match = np.zeros((match_width))
        for x in range(match_width):
            roi = img_sub_cross_lumi[:, x:x+width_cross]
            match[x] = 1 - sum(roi[mask_cross]) / cross_px / 255
        # print(str.format("time matching:{0}sec", time.perf_counter() - time_start))

        # plt.plot(match)
        # plt.show()
        # cv2.imshow('roi_lamps', roi_lamps[::shrink_rate, ::shrink_rate])
        # cv2.waitKey(0)

        num_cross = 0
        lamps = [1, 1, 1, 1]
        for i in range(self.num_lumps_per_team):
            # plt.plot(match)
            # plt.show()
            x_max = np.argmax(match)
            if (match[x_max] >= self.th_cross_match):
                num_cross += 1
                lamps[self.IconSenterXtoIndex(x_max * shrink_rate, opponent)] = 0 # デス
                match[max(0, x_max-span_lamps//2):min(x_max+span_lamps//2, match.shape[0])] = 0
            else:
                break

        # Search offline lamps
        top_offline = (roi_lamps.shape[0] - self.height_offline_template) // 2
        roi_offline = roi_lamps[top_offline:top_offline+self.height_offline_template, :]
        offline_match = 1 - cv2.matchTemplate(roi_offline, self.img_offline_template, cv2.TM_SQDIFF)[0] / self.height_offline_template / self.width_offline_template / (255 * 255)

        for i in range(self.num_lumps_per_team):
            x_max = np.argmax(offline_match)
            # print("offline_match[x_max]: {0}".format(offline_match[x_max]))
            # plt.plot(offline_match)
            # cv2.imshow('roi_offline', roi_offline)
            # plt.show()
            # cv2.waitKey(0)

            if (offline_match[x_max] >= self.th_offline_match):
                # print("offline_match[x_max]: {0}".format(offline_match[x_max]))
                # plt.plot(offline_match)
                # plt.show()
                # cv2.imshow('roi_offline', roi_offline)
                # cv2.waitKey(1)

                lamps[self.IconSenterXtoIndex(x_max - 8, opponent)] = -1 # 切断 -8は、×のテンプレート幅48pxに対しオフラインテンプレート幅が32pxであるため座標を合わせる
                offline_match[max(0, x_max-self.span_lamps//2):min(x_max+self.span_lamps//2, offline_match.shape[0])] = 0
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
    th_digit_similarity = 0.7
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
    # num_buffer_frame_penalty = 5

    count_same_our_penalty = 0
    latest_our_penalty = 0
    count_same_opponent_penalty = 0
    latest_opponent_penalty = 0
    th_count_same_penalty_not_zero = 3
    th_count_same_penalty_zero = 30
    th_binarize = 243
    
    def __init__(self, path_template_count100):
        img = cv2.imread(path_template_count100)
        self.img_template_count100 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 数字のテンプレートを読み込む
        self.img_template_digit = []
        for i in range(10):
            self.img_template_digit.append(cv2.cvtColor(cv2.imread(r".\template\digit_" + str(i) + ".png"), cv2.COLOR_BGR2GRAY))
        self.digit_height, self.digit_width = self.img_template_digit[0].shape[:2]

        # ペナルティを読んだ結果のバッファ
        # self.our_pnealty_buffer = [-1 for x in range(self.num_buffer_frame_penalty)]
        # self.opponent_pnealty_buffer = [-1 for x in range(self.num_buffer_frame_penalty)]

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
        cv2.imshow('our_count', img_bin)
        cv2.waitKey(1)
        
        # 100とマッチング
        match = cv2.matchTemplate(img_bin, self.img_template_count100, cv2.TM_CCOEFF_NORMED)
        if np.max(match) >= self.th_100match:
            count_self = 100
        else:
            # 二値化の方式を変えて再び100とマッチング
            img_bin_alt = self.rotateAndBinarize(roi, 5, self.th_binarize)
            match = cv2.matchTemplate(img_bin_alt, self.img_template_count100, cv2.TM_CCOEFF_NORMED)

            # cv2.imshow('our_count', img_bin_alt)
            # cv2.waitKey(1)

            if np.max(match) >= self.th_100match:
                count_self = 100
            else:
                # 縮小して数字テンプレートのサイズに合わせる
                img_resize = cv2.resize(img_bin, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
                # cv2.imshow('our_count self', img_resize)
                # cv2.waitKey(1)

                # 数字読み
                count_self = self.readTwoDigitsOld(img_resize)

                # 失敗したときは、固定値二値化で再チャレンジ
                if count_self < 0:
                    img_bin = self.rotateAndBinarize(roi, 5, self.th_binarize)
                    img_resize = cv2.resize(img_bin, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
                    count_self = self.readTwoDigitsOld(img_resize)
                    
                    # cv2.imshow('our_count', img_bin)
                    # cv2.waitKey(1)
        
        # 相手
        roi = img_capture[150:210, 1035:1115]
        img_bin = self.rotateAndBinarize(roi, -5)

        # 100とマッチング
        match = cv2.matchTemplate(img_bin, self.img_template_count100, cv2.TM_CCOEFF_NORMED)
        if np.max(match) >= self.th_100match:
            count_opponent = 100
        else:
            # 二値化の方式を変えて再び100とマッチング
            img_bin_alt = self.rotateAndBinarize(roi, -5, self.th_binarize)
            match = cv2.matchTemplate(img_bin_alt, self.img_template_count100, cv2.TM_CCOEFF_NORMED)

            # cv2.imshow('our_count', img_bin_alt)
            # cv2.waitKey(1)

            if np.max(match) >= self.th_100match:
                count_opponent = 100
            else:
                # 縮小して数字テンプレートのサイズに合わせる
                img_resize = cv2.resize(img_bin, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)

                # 数字読み
                count_opponent = self.readTwoDigitsOld(img_resize)

                # 失敗したときは、固定値二値化で再チャレンジ
                if count_opponent < 0:
                    img_bin = self.rotateAndBinarize(roi, -5, self.th_binarize)
                    img_resize = cv2.resize(img_bin, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
                    count_opponent = self.readTwoDigitsOld(img_resize)

                    # cv2.imshow('count opponent', img_resize)

        # cv2.waitKey(1)

        return (count_self, count_opponent)

    def rotateAndBinarize(self, img, degree, th_lumi=None):
        # グレースケール化
        roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 回転
        h, w = roi_gray.shape[:2]
        center = (int(w / 2), int(h / 2))
        trans = cv2.getRotationMatrix2D(center, degree, 1.0)
        img_rotate = cv2.warpAffine(roi_gray, trans, (w,h))

        if th_lumi != None:
            # 指定値で二値化
            ret, img_binarize = cv2.threshold(img_rotate, th_lumi, 255, cv2.THRESH_BINARY)
        else:
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
        # print(str.format("digit:1 match:{0}", result[1]))
        # print(str.format("digit:7 match:{0}", result[7]))
        return np.argmax(result) if np.max(result) >= th_digit_match else -1

    def readAreaPenalty(self, img_capture):
        # 味方 
        roi_our_penalty = img_capture[self.roi_penalty_top:self.roi_penalty_bottom, self.roi_our_penalty_left:self.roi_our_penalty_left+self.roi_penalty_width]
        gray_our_penalty = cv2.cvtColor(roi_our_penalty, cv2.COLOR_BGR2GRAY)
        img_our_penalty_resized = cv2.resize(gray_our_penalty, dsize=(self.digit_width*2, self.digit_height), interpolation=cv2.INTER_NEAREST)
        ret, img_otsu = cv2.threshold(img_our_penalty_resized, 0, 255, cv2.THRESH_OTSU)
        our_pnealty = self.readTwoDigits(img_otsu, 0.65)

        # 同じ数値の継続判定
        if self.latest_our_penalty == our_pnealty:
            self.count_same_our_penalty += 1
        else:
            self.latest_our_penalty = our_pnealty
            self.count_same_our_penalty = 1

        # カウントの確定
        if self.latest_our_penalty > 0 and self.count_same_our_penalty < self.th_count_same_penalty_not_zero:
            our_pnealty = -1 # まだ確かでない→読めず
        if self.latest_our_penalty < 0 and self.count_same_our_penalty >= self.th_count_same_penalty_zero:
            our_pnealty = 0 # 長い期間読めず→非表示(ゼロ)と判断

        # print("our_pnealty:" + str(our_pnealty))
        # cv2.imshow('OurPenalty', img_otsu)
        # cv2.waitKey(1)
        
        # 敵
        roi_opponent_penalty = img_capture[self.roi_penalty_top:self.roi_penalty_bottom, self.roi_opponent_penalty_left:self.roi_opponent_penalty_left+self.roi_penalty_width]
        gray_opponent_penalty = cv2.cvtColor(roi_opponent_penalty, cv2.COLOR_BGR2GRAY)
        img_opponent_penalty_resized = cv2.resize(gray_opponent_penalty, dsize=(self.digit_width*2, self.digit_height), interpolation=cv2.INTER_NEAREST)
        ret, img_otsu = cv2.threshold(img_opponent_penalty_resized, 0, 255, cv2.THRESH_OTSU)
        opponent_pnealty = self.readTwoDigits(img_otsu, 0.65)

        # 同じ数値の継続判定
        if self.latest_opponent_penalty == opponent_pnealty:
            self.count_same_opponent_penalty += 1
        else:
            self.latest_opponent_penalty = opponent_pnealty
            self.count_same_opponent_penalty = 1

        # カウントの確定
        if self.latest_opponent_penalty > 0 and self.count_same_opponent_penalty < self.th_count_same_penalty_not_zero:
            opponent_pnealty = -1 # まだ確かでない→読めず
        if self.latest_opponent_penalty < 0 and self.count_same_opponent_penalty >= self.th_count_same_penalty_zero:
            opponent_pnealty = 0 # 長い期間読めず→非表示(ゼロ)と判断

        return our_pnealty, opponent_pnealty

    def readTwoDigits(self, img_digits, th_digit_match = None): #後にこちらへ統一したい
        th_digit_match = self.th_digit_match if th_digit_match == None else th_digit_match

        # 画像サイズのチェック
        if img_digits.shape[0] != self.digit_height or img_digits.shape[1] != self.digit_width * 2:
            return -1

        # ROIの上下右端に白画素がある場合、誤読リスクがあるので読まない（左は1桁のときの+があるため適さない）
        # if np.sum(img_digits[:1, :]) + np.sum(img_digits[-1:, :]) + np.sum(img_digits[:, :1]) + np.sum(img_digits[:, -1:]) > 0:
        if np.sum(img_digits[:1, :]) + np.sum(img_digits[-1:, :]) + np.sum(img_digits[:, -1:]) > 0:
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


class FinishSearcher:
    roi_top = 640
    roi_bottom = 740
    roi_left = 1000
    roi_right = 1250

    def __init__(self, template_img_path, th_finish_match = 0.9):
        self.img_template = cv2.cvtColor(cv2.imread(template_img_path), cv2.COLOR_BGR2GRAY)
        self.th_finish_match = th_finish_match

    def find(self, img_capture):
        roi = img_capture[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right] 
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

        result = cv2.matchTemplate(img_otsu, self.img_template, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        # print(f"max value: {maxVal}, position: {maxLoc}")
        # cv2.imshow('img_otsu', img_otsu)
        # cv2.waitKey(0)

        if maxVal >= self.th_finish_match:
            return True
        else:
            return False


class Judge(Enum):
    none = 0
    win = 1
    lose = 2

class JudgeSearcher:
    roi_top = 40
    roi_bottom = 150
    roi_left = 0
    roi_right = 250

    def __init__(self, template_win_img_path, template_lose_img_path, th_match = 0.9):
        self.win_img_template = cv2.cvtColor(cv2.imread(template_win_img_path), cv2.COLOR_BGR2GRAY)
        self.lose_img_template = cv2.cvtColor(cv2.imread(template_lose_img_path), cv2.COLOR_BGR2GRAY)
        self.th_match = th_match

    def find(self, img_capture):
        roi = img_capture[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right] 
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

        # cv2.imshow('img_otsu', img_otsu)
        # cv2.waitKey(0)

        result = cv2.matchTemplate(img_otsu, self.win_img_template, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        if maxVal >= self.th_match:
            return Judge.win

        result = cv2.matchTemplate(img_otsu, self.lose_img_template, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        if maxVal >= self.th_match:
            return Judge.lose

        return Judge.none
