import cv2
import moviepy.editor
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import recognition
import utility
import time
import glob


default_output_dir = r"D:\Documents\Python\splatoon\output"
default_movie_path = r'D:\Documents\OBS\ガチエリア\2022-10-31_12-27-05.mp4'
# default_movie_path = r'D:\Documents\OBS\ガチエリア\回線落ち_要対処\2022-11-30_21-45-44_開始直後の無効試合.mp4'
# default_movie_path = r'D:\Documents\OBS\2022-10-11_21-47-06_キルデスシーン.mp4'

# 動画出力
def clipMovie(path_movie, path_seve, clip_start_sec, clip_end_sec, fps=30):
    video = moviepy.editor.VideoFileClip(path_movie).subclip(clip_start_sec, clip_end_sec)
    video.write_videofile(path_seve, fps=fps)


def main(movie_path, output_dir, killdeath_movie=False, recognition_movie=False):
    if not os.path.exists(movie_path):
        print("File is not exist: " + movie_path)
        return

    # 動画キャプチャ
    cap = cv2.VideoCapture(movie_path)
    video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
    video_fps = cap.get(cv2.CAP_PROP_FPS)                 # フレームレートを取得する
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 189 * video_fps) # デバッグ用フレーム移動

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
    finish_searcher = recognition.FinishSearcher(r".\template\finish.png")
    judge_searcher = recognition.JudgeSearcher(r".\template\win.png", r".\template\lose.png")
    interruption_searcher = recognition.InterruptionSearcher(r".\template\invalid_match.png", r".\template\connection_error.png")

    # ゲーム時間1秒単位のタイムライン
    game_time_max = int(video_frame_count / video_fps) + 300 # 最短300秒+動画時間
    cols = ["Rule", "Result", "GameTime[s]", "OurCount", "OpponentCount", "OurPenalty", "OpponentPenalty", "OurNumLive", "OpponentNumLive",
            "Kill", "Death", "NumSubLive", "Initiative",
            "LampOur1", "LampOur2", "LampOur3", "LampOur4", "LampOpponent1", "LampOpponent2", "LampOpponent3", "LampOpponent4"]
    df_timeline = pd.DataFrame([["-", "-", 0, 100, 100, 0, 0, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], columns=cols) # ゲーム開始時
    for i in range(1, game_time_max):
        # -1は読み取れなかった場合に残る初期値
        df_timeline = df_timeline.append(pd.Series(["-", "-", i, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -2, -2, -2, -2, -2, -2, -2, -2], index=df_timeline.columns), ignore_index=True)

    # 認識結果movie
    if recognition_movie:
        recognition_movie_maker = utility.RecognitionMovieMaker(
            os.path.join(output_dir, os.path.basename(movie_path).split(".")[0] + "_recognition.mp4"),
            30, (video_width, video_height)
        )

    # 初期化
    index_frame = 1
    prev_last_time_sec = -1
    num_kill_prev = 0
    events = []
    gachi_kind = recognition.GachiKind.none
    our_lamp = [1, 1, 1, 1]
    opponent_lamp = [1, 1, 1, 1]
    count = [100, 100]
    penalty  = [0, 0]
    time_finish = -1
    proc_time = {}
    read_frecuency = 5
    judge = recognition.Judge.none

    while True:
        # フレーム情報取得
        time_start = time.perf_counter()
        ret, img_frame = cap.read()
        proc_time["time_get_a_frame"] = time.perf_counter() - time_start

        # 動画が終われば処理終了
        if ret == False:
            break    

        # img_frame = cv2.imread(r"D:\Documents\OBS\area_count_cannot_read.png") #debug

        # 残り時間読み
        time_start = time.perf_counter()
        last_sec = time_reader.read(img_frame) # -1で延長中、Noneで時間非表示（ゲーム前後かマップ表示中）
        indicator_enabled = True if last_sec != None else False

        # 1秒減ることを観測したら時刻アンカー
        if indicator_enabled:
            if last_sec == prev_last_time_sec - 1:
                time_manager.setAnchor(last_sec, index_frame)
            prev_last_time_sec = last_sec
        game_time = time_manager.frameToGameTime(index_frame)
        game_time_int = int(game_time)
        proc_time["time_read_time"] = time.perf_counter() - time_start

        # 各種インジケータが表示中であれば読む
        num_kill = None
        death = None
        if indicator_enabled:
            # カウント読み(毎フレーム読む)
            time_start = time.perf_counter()
            gachi_kind, count, penalty = count_reader.read(img_frame)
            proc_time["time_read_count"] = time.perf_counter() - time_start

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

            # 所定フレームごとに読む
            if index_frame % read_frecuency == 0:
                # イカランプ読み
                time_start = time.perf_counter()
                our_lamp, opponent_lamp = ika_lamp_reader.read(img_frame)
                proc_time["time_read_lamps"] = time.perf_counter() - time_start

                if game_time_int > 0:
                    df_timeline.loc[game_time_int, "OurNumLive"] = our_lamp.count(1)
                    df_timeline.loc[game_time_int, "OpponentNumLive"] = opponent_lamp.count(1)
                    df_timeline.loc[game_time_int, "LampOur1"] = our_lamp[0]
                    df_timeline.loc[game_time_int, "LampOur2"] = our_lamp[1]
                    df_timeline.loc[game_time_int, "LampOur3"] = our_lamp[2]
                    df_timeline.loc[game_time_int, "LampOur4"] = our_lamp[3]
                    df_timeline.loc[game_time_int, "LampOpponent1"] = opponent_lamp[0]
                    df_timeline.loc[game_time_int, "LampOpponent2"] = opponent_lamp[1]
                    df_timeline.loc[game_time_int, "LampOpponent3"] = opponent_lamp[2]
                    df_timeline.loc[game_time_int, "LampOpponent4"] = opponent_lamp[3]                    
            
                # やられたサーチ
                time_start = time.perf_counter()
                death = death_searcher.find(img_frame, index_frame)
                if death:
                    events.append(utility.Event(utility.EventKind.death, index_frame))
                proc_time["time_find_death"] = time.perf_counter() - time_start
                
                # たおしたサーチ
                time_start = time.perf_counter()
                num_kill = kill_searcher.find(img_frame)
                if num_kill > num_kill_prev:
                    events.append(utility.Event(utility.EventKind.kill, index_frame))
                num_kill_prev = num_kill
                proc_time["time_find_kill"] = time.perf_counter() - time_start

        elif index_frame > 0 and game_time_int > 0:
            if time_finish < 0:
                # インジケータが非表示の場合、Finish読み
                time_start = time.perf_counter()
                if finish_searcher.find(img_frame):
                    time_finish = game_time_int
                    # break
                proc_time["time_find_finish"] = time.perf_counter() - time_start

                # 中断読み
                if interruption_searcher.find(img_frame):
                    time_finish = game_time_int - 1 # 中断となってから認識までのラグ対策に-1
                    break
            else:
                # Finish読み後はWIN/LOSE読み
                judge = judge_searcher.find(img_frame)
                if judge != recognition.Judge.none:
                    print("Judge: {0}".format(judge))
                    break

        # 認識結果movie
        if recognition_movie:
            recognition_movie_maker.makeFrame(img_frame, 300 - game_time_int, our_lamp, opponent_lamp, count, penalty, num_kill, death)

        # debug:状態表示
        time_start = time.perf_counter()
        print(str.format("Frame:{0}, GameTime:{1:0.2f}sec LastTime:{2}sec gachi_kind:{3}, lamp:{4}vs{5}, count:{6} penalty:{7}",
            index_frame, game_time, last_sec if last_sec != None else "-", gachi_kind, our_lamp, opponent_lamp, count, penalty))
        # print(proc_time)
        cv2.imshow('Video', img_frame[:270,450:-450])
        cv2.waitKey(1)
        proc_time["time_debug"] = time.perf_counter() - time_start

        index_frame += 1


    cap.release()
    cv2.destroyAllWindows()

    # 認識結果movie
    if recognition_movie:
        recognition_movie_maker.release()

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
        if df_timeline.loc[i, "LampOur1"] < -1:
            df_timeline.loc[i, "LampOur1"] = df_timeline.loc[i - 1, "LampOur1"]
        if df_timeline.loc[i, "LampOur2"] < -1:
            df_timeline.loc[i, "LampOur2"] = df_timeline.loc[i - 1, "LampOur2"]
        if df_timeline.loc[i, "LampOur3"] < -1:
            df_timeline.loc[i, "LampOur3"] = df_timeline.loc[i - 1, "LampOur3"]
        if df_timeline.loc[i, "LampOur4"] < -1:
            df_timeline.loc[i, "LampOur4"] = df_timeline.loc[i - 1, "LampOur4"]
        if df_timeline.loc[i, "LampOpponent1"] < -1:
            df_timeline.loc[i, "LampOpponent1"] = df_timeline.loc[i - 1, "LampOpponent1"]
        if df_timeline.loc[i, "LampOpponent2"] < -1:
            df_timeline.loc[i, "LampOpponent2"] = df_timeline.loc[i - 1, "LampOpponent2"]
        if df_timeline.loc[i, "LampOpponent3"] < -1:
            df_timeline.loc[i, "LampOpponent3"] = df_timeline.loc[i - 1, "LampOpponent3"]
        if df_timeline.loc[i, "LampOpponent4"] < -1:
            df_timeline.loc[i, "LampOpponent4"] = df_timeline.loc[i - 1, "LampOpponent4"]

        # 分析用にデータ加工
        df_timeline.loc[i, "NumSubLive"] = df_timeline.loc[i, "OurNumLive"] - df_timeline.loc[i, "OpponentNumLive"]
        
        if gachi_kind == recognition.GachiKind.area:
            # ガチエリアでの主導権はカウント進行で把握
            if df_timeline.loc[i - 1, "OurCount"] - df_timeline.loc[i, "OurCount"] > 0 or df_timeline.loc[i - 1, "OurPenalty"] - df_timeline.loc[i, "OurPenalty"] > 0:
                df_timeline.loc[i, "Initiative"] = 1
            elif df_timeline.loc[i - 1, "OpponentCount"] - df_timeline.loc[i, "OpponentCount"] > 0 or df_timeline.loc[i - 1, "OpponentPenalty"] - df_timeline.loc[i, "OpponentPenalty"] > 0:
                df_timeline.loc[i, "Initiative"] = -1
            else:
                df_timeline.loc[i, "Initiative"] = 0

    # 勝敗をセット
    if judge != recognition.Judge.none:
        for i in range(game_time_max):
            df_timeline.loc[i, "Result"] = "WIN" if judge == recognition.Judge.win else "LOSE"

    # ルールをセット：現在はエリア固定
    for i in range(game_time_max):
        df_timeline.loc[i, "Rule"] = "Area"

    # Finishを読めていた場合、その時間でタイムラインを打ち切り
    if time_finish >= 0:
        for i in range(time_finish + 1, game_time_max):
            df_timeline = df_timeline.drop(index=i)

    output_csv_path = os.path.join(output_dir, os.path.basename(movie_path).split(".")[0] + "_timeline.csv")
    df_timeline.to_csv(output_csv_path)

    # キルデス部分の切り出し動画
    if killdeath_movie:
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
    main(default_movie_path, default_output_dir, recognition_movie=False)
   
    # movie_paths = glob.glob(r"D:\Documents\OBS\ガチエリア\回線落ち_要対処\*.mp4")
    # for path in movie_paths:
    #     if 0 < len(glob.glob(os.path.join("D:\Documents\Python\splatoon\output", os.path.basename(path).split(".")[0] + "*.csv"))):
    #         continue
    #     main(path, default_output_dir)
