import cv2
import moviepy.editor
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import recognition
import utility
       

default_output_dir = r"D:\Documents\Python\splatoon\output"
default_movie_path = r'D:\Documents\OBS\test2.mp4'
default_movie_path = r'D:\Documents\OBS\ガチエリア\2022-10-12_19-27-37_エリア勝利_キンメダイ美術館.mp4' #エリアノックアウト勝利
# default_movie_path = r'D:\Documents\OBS\ガチエリア\2022-10-11_21-29-17_昇格戦_エリア敗北_海女美術大学.mp4' #エリアノックアウト敗北
default_movie_path = r'D:\Documents\OBS\ガチエリア\2022-10-31_12-33-02_延長突入.mp4' #延長突入
# default_movie_path = r'D:\Documents\OBS\ガチエリア\ペナルティ読みミス.mp4'

# 動画出力
def clipMovie(path_movie, path_seve, clip_start_sec, clip_end_sec, fps=30):
    video = moviepy.editor.VideoFileClip(path_movie).subclip(clip_start_sec, clip_end_sec)
    video.write_videofile(path_seve, fps=fps)


def main(movie_path, output_dir):
    # 動画キャプチャ
    cap = cv2.VideoCapture(movie_path)
    video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
    video_fps = cap.get(cv2.CAP_PROP_FPS)                 # フレームレートを取得する

    # 各種画像処理エンジン
    time_manager = utility.TimeManager(video_fps)
    kill_searcher = recognition.KillSearcher(r".\template\taoshita.png")
    death_searcher = recognition.DeathSearcher(r".\template\yarareta.png")
    count_reader = recognition.CountReader(r".\template\area100.png")
    ika_lamp_reader = recognition.IkaLampReader(r".\template\lamp_cross.png", r".\template\lamp_offline.png")
    digit_template_paths = []
    for i in range(10):
        digit_template_paths.append(r".\template\digit_" + str(i) + ".png")
    time_reader = recognition.TimeReader(digit_template_paths, r".\template\extra_time.png")

    # ゲーム時間1秒単位のタイムライン
    game_time_max = int(video_frame_count / video_fps) + 300 # 最短300秒+動画時間
    cols = ["GameTime[s]", "OurCount", "OpponentCount", "OurPenalty", "OpponentPenalty", "OurNumLive", "OpponentNumLive", "Kill", "Death"]
    df_timeline = pd.DataFrame([[0, 100, 100, 0, 0, 4, 4, 0, 0]], columns=cols) # ゲーム開始時
    for i in range(1, game_time_max):
        # -1は読み取れなかった場合に残る初期値
        df_timeline = df_timeline.append(pd.Series([i, -1, -1, -1, -1, -1, -1, 0, 0], index=df_timeline.columns), ignore_index=True)

    # 初期化
    index_frame = 1
    prev_last_time_sec = -1
    num_kill_prev = 0
    events = []
    gachi_kind = recognition.GachiKind.none
    our_lamp = [True, True, True, True]
    opponent_lamp = [True, True, True, True]
    count = [100, 100]
    penalty  = [0, 0]

    while True:
        # フレーム情報取得
        ret, img_frame = cap.read()
        
        # 動画が終われば処理終了
        if ret == False:
            break    

        # img_frame = cv2.imread(r"D:\Documents\OBS\map.png") #debug

        # 残り時間読み
        last_sec = time_reader.read(img_frame) # -1で延長中、Noneで時間非表示（ゲーム前後かマップ表示中）
        none_indicator = True if last_sec == None else False

        # 1秒減ることを観測したら時刻アンカー
        if not none_indicator:
            if last_sec == prev_last_time_sec - 1:
                time_manager.setAnchor(last_sec, index_frame)
            prev_last_time_sec = last_sec
        game_time = time_manager.frameToGameTime(index_frame)
        game_time_int = int(game_time)

        # 各種インジケータが表示中であれば読む
        if not none_indicator:
            # やられたサーチ
            if death_searcher.find(img_frame, index_frame):
                events.append(utility.Event(utility.EventKind.death, index_frame))
            
            # たおしたサーチ
            num_kill = kill_searcher.find(img_frame)
            if num_kill > num_kill_prev:
                events.append(utility.Event(utility.EventKind.kill, index_frame))
            num_kill_prev = num_kill

            # カウント読み
            gachi_kind, count, penalty = count_reader.read(img_frame)

            # イカランプ読み
            our_lamp, opponent_lamp = ika_lamp_reader.read(img_frame)

            # 読んだ結果の格納
            if game_time_int > 0:
                if count[0] >= 0:
                    df_timeline.loc[game_time_int, "OurCount"] = count[0]
                if count[1] >= 0:
                    df_timeline.loc[game_time_int, "OpponentCount"] = count[1]
                if penalty[0] >= 0:
                    df_timeline.loc[game_time_int, "OurPenalty"] = penalty[0]
                if penalty[1] >= 0:
                    df_timeline.loc[game_time_int, "OpponentPenalty"] = penalty[1]
                if np.sum(our_lamp) >= 0:
                    df_timeline.loc[game_time_int, "OurNumLive"] = np.sum(our_lamp)
                if np.sum(opponent_lamp) >= 0:
                    df_timeline.loc[game_time_int, "OpponentNumLive"] = np.sum(opponent_lamp)
        
        elif index_frame > 0 and game_time_int > 0:
            # インジケータが非表示の場合、情報の引継ぎ
            df_timeline.loc[game_time_int, "OurCount"] = df_timeline.loc[game_time_int - 1, "OurCount"]
            df_timeline.loc[game_time_int, "OpponentCount"] = df_timeline.loc[game_time_int - 1, "OpponentCount"]
            df_timeline.loc[game_time_int, "OurNumLive"] = df_timeline.loc[game_time_int - 1, "OurNumLive"]
            df_timeline.loc[game_time_int, "OpponentNumLive"] = df_timeline.loc[game_time_int - 1, "OpponentNumLive"]

        # debug:状態表示
        print(str.format("Frame:{0}, GameTime:{1:0.2f}sec LastTime:{2}sec gachi_kind:{3}, lamp:{4}vs{5}, count:{6} penalty:{7}",
            index_frame, game_time, last_sec if last_sec != None else "-", gachi_kind, our_lamp, opponent_lamp, count, penalty))
        cv2.imshow('Video', img_frame[:270,450:-450])
        cv2.waitKey(1)

        index_frame += 1


    cap.release()
    cv2.destroyAllWindows()

    # return #debug

    # CSV出力
    cols = ["GameTime[s]", "Event"]
    df_event_log = pd.DataFrame([], columns=cols)
    for item in events:
        data = [str.format("{0:0.2f}", time_manager.frameToGameTime(item.frame))]
        if item.event_kind == utility.EventKind.death:
            data.append("death")
        elif item.event_kind == utility.EventKind.kill:
            data.append("kill")
        df_event_log = df_event_log.append(pd.Series(data, index=df_event_log.columns), ignore_index=True)
    output_csv_path = os.path.join(output_dir, os.path.basename(movie_path).split(".")[0] + "_event.csv")
    df_event_log.to_csv(output_csv_path)

    # 1秒単位の時系列
    for item in events:
        time_sec = int(time_manager.frameToGameTime(item.frame))
        if item.event_kind == utility.EventKind.death:
            df_timeline.loc[time_sec, "Death"] += 1
        elif item.event_kind == utility.EventKind.kill:
            df_timeline.loc[time_sec, "Kill"] += 1
    for i in range(1, game_time_max):
        df_timeline.loc[i, "Death"] += df_timeline.loc[i - 1, "Death"]
        df_timeline.loc[i, "Kill"] += df_timeline.loc[i - 1, "Kill"]
        if df_timeline.loc[i, "OurCount"] < 0:
            df_timeline.loc[i, "OurCount"] = df_timeline.loc[i - 1, "OurCount"]
        if df_timeline.loc[i, "OpponentCount"] < 0:
            df_timeline.loc[i, "OpponentCount"] = df_timeline.loc[i - 1, "OpponentCount"]
        if df_timeline.loc[i, "OurPenalty"] < 0:
            df_timeline.loc[i, "OurPenalty"] = df_timeline.loc[i - 1, "OurPenalty"]
        if df_timeline.loc[i, "OpponentPenalty"] < 0:
            df_timeline.loc[i, "OpponentPenalty"] = df_timeline.loc[i - 1, "OpponentPenalty"]
        if df_timeline.loc[i, "OurNumLive"] < 0:
            df_timeline.loc[i, "OurNumLive"] = df_timeline.loc[i - 1, "OurNumLive"]
        if df_timeline.loc[i, "OpponentNumLive"] < 0:
            df_timeline.loc[i, "OpponentNumLive"] = df_timeline.loc[i - 1, "OpponentNumLive"]

    output_csv_path = os.path.join(output_dir, os.path.basename(movie_path).split(".")[0] + "_timeline.csv")
    df_timeline.to_csv(output_csv_path)

    death_count = 0
    kill_count = 0
    for item in events:
        if item.event_kind == utility.EventKind.death:
            clip_start = max(item.frame / video_fps - 10, 0)
            clip_end = min(item.frame / video_fps + 3, video_frame_count / video_fps - 1)
            save_path = os.path.join(output_dir, str.format("{0}_death_{1:04d}.mp4", os.path.basename(movie_path).split(".")[0], death_count))
            clipMovie(movie_path, save_path, clip_start, clip_end)
            death_count += 1
        elif item.event_kind == utility.EventKind.kill:
            clip_start = max(item.frame / video_fps - 10, 0)
            clip_end = min(item.frame / video_fps + 5, video_frame_count / video_fps - 1)
            save_path = os.path.join(output_dir, str.format("{0}_kill_{1:04d}.mp4", os.path.basename(movie_path).split(".")[0], kill_count))
            clipMovie(movie_path, save_path, clip_start, clip_end)
            kill_count += 1


if __name__ == "__main__":
    main(default_movie_path, default_output_dir)
