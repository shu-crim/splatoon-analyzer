import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import datetime

default_input_dir = r"D:\Documents\Python\splatoon\output"
default_output_dir = r"D:\Documents\Python\splatoon\output\analyze"

def main(input_dir, output_dir):
    timeline_csv_paths = glob.glob(os.path.join(default_input_dir, "*timeline.csv"))

    num_game = len(timeline_csv_paths)
    num_win = 0
    num_lose = 0

    num_first_initiative = 0
    num_first_initiative_and_win = 0
    num_lost_first_initiative_and_win = 0
    num_lost_first_initiative_and_lose = 0

    num_first_kill = 0
    num_first_kill_and_win = 0
    num_lost_first_kill_and_win = 0
    num_lost_first_kill_and_lose = 0

    num_first_2sub = 0
    num_first_2sub_and_win = 0
    num_lost_first_2sub = 0
    num_lost_first_2sub_and_win = 0
    num_lost_first_2sub_and_lose = 0

    filename_head = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    cols = ["ファイル名", "勝利チーム[自1/0/-1敵]", "試合時間[s]", "自チーム総キル数", "敵チーム総キル数",
        "自チームキル速度レート[kill/min]", "敵チームキルレート[kill/min]", "勝利チームキルレート[kill/min]", "勝利チームキルレート差[kill/min]",
        "ペナルティ無し仮定での勝利チーム[1/0/-1]", "ペナルティ無し仮定での試合時間[s]", "ペナルティ無し仮定では逆転[逆転1/0そのまま]"]
    df_stats = pd.DataFrame([], columns=cols)

    with open(os.path.join(output_dir, filename_head + "_count_analysis.csv"), mode='w') as f:
        f.write("ファイル名,時刻[s],人数差,その後10秒でのカウント増減,その後20秒でのカウント増減,その後30秒でのカウント増減\n")

    num_input_csv = 0
    first_df = True
    for path in timeline_csv_paths:
        new_df = pd.read_csv(path)

        # 勝利したか
        win = False
        lose = False
        if new_df.loc[0, "Result"] == "WIN":
            win = True
            num_win += 1
        elif new_df.loc[0, "Result"] == "LOSE":
            lose = True
            num_lose += 1

        # カウント先制したか
        first_initiative = False
        for i in range(len(new_df)):
            if new_df.loc[i, "Initiative"] > 0:
                first_initiative = True
                break
            elif new_df.loc[i, "Initiative"] < 0:
                break
        if first_initiative:
            num_first_initiative += 1
            if win:
                num_first_initiative_and_win += 1
        else:
            if win:
                num_lost_first_initiative_and_win += 1
            elif lose:
                num_lost_first_initiative_and_lose += 1

        # 先制キルしたか
        first_kill = False
        for i in range(len(new_df)):
            if new_df.loc[i, "NumSubLive"] > 0:
                first_kill = True
                break
            elif new_df.loc[i, "NumSubLive"] < 0:
                break
        if first_kill:
            num_first_kill += 1
            if win:
                num_first_kill_and_win += 1
        else:
            if win:
                num_lost_first_kill_and_win += 1
            elif lose:
                num_lost_first_kill_and_lose += 1

        # 先に2人差を取ったか
        got_first_2sub = False
        lost_first_2sub = False
        for i in range(len(new_df)):
            if new_df.loc[i, "NumSubLive"] >= 2:
                got_first_2sub = True
                break
            elif new_df.loc[i, "NumSubLive"] <= -2:
                lost_first_2sub = True
                break
        if got_first_2sub:
            num_first_2sub += 1
            if win:
                num_first_2sub_and_win += 1
        if lost_first_2sub:
            num_lost_first_2sub += 1
            if win:
                num_lost_first_2sub_and_win += 1
            if lose:
                num_lost_first_2sub_and_lose += 1

        # ペナルティ消化も含めたカウント進行量を配列化
        our_count_progress = np.zeros((len(new_df)), int)
        opponent_count_progress = np.zeros((len(new_df)), int)
        total_our_count = np.zeros((len(new_df)), int)
        total_opponent_count = np.zeros((len(new_df)), int)
        total_sub_count = np.zeros((len(new_df)), int)
        for i in range(1, len(new_df)):
            our_count_progress[i] += (new_df.loc[i - 1, "OurCount"] - new_df.loc[i, "OurCount"])
            if new_df.loc[i - 1, "OurPenalty"] > new_df.loc[i, "OurPenalty"]:
                our_count_progress[i] += (new_df.loc[i - 1, "OurPenalty"] - new_df.loc[i, "OurPenalty"])
            opponent_count_progress[i] += (new_df.loc[i - 1, "OpponentCount"] - new_df.loc[i, "OpponentCount"])                
            if new_df.loc[i - 1, "OpponentPenalty"] > new_df.loc[i, "OpponentPenalty"]:
                opponent_count_progress[i] += (new_df.loc[i - 1, "OpponentPenalty"] - new_df.loc[i, "OpponentPenalty"])
            total_our_count[i] = our_count_progress[i] + total_our_count[i - 1]
            total_opponent_count[i] = opponent_count_progress[i] + total_opponent_count[i - 1]
            total_sub_count[i] = total_sub_count[i - 1] + our_count_progress[i] - opponent_count_progress[i]

        # もしペナルティが無かった場合の勝利チームを判定
        win_team_without_penalty = 0
        game_time_without_penalty = 0
        if win or lose:
            our_count = 100
            opponent_count = 100
            for i in range(1, len(new_df)):
                our_count -= our_count_progress[i]
                opponent_count -= opponent_count_progress[i]            

                if our_count <= 0 and opponent_count > 0:
                    win_team_without_penalty = 1
                    game_time_without_penalty = i
                    break
                elif our_count > 0 and opponent_count <= 0:
                    win_team_without_penalty = -1
                    game_time_without_penalty = i
                    break
        
        # 人数差が増加したときのカウント進行差を算出
        with open(os.path.join(output_dir, filename_head + "_count_analysis.csv"), mode='a') as f:
            for i in range(1, len(new_df)):
                count_after_10s = total_sub_count[i + 10] - total_sub_count[i] if i + 10 < len(new_df) else "-"
                count_after_20s = total_sub_count[i + 20] - total_sub_count[i] if i + 20 < len(new_df) else "-"
                count_after_30s = total_sub_count[i + 30] - total_sub_count[i] if i + 30 < len(new_df) else "-"

                if (new_df.loc[i, "NumSubLive"] > 0 and new_df.loc[i, "NumSubLive"] > new_df.loc[i - 1, "NumSubLive"]) or \
                   (new_df.loc[i, "NumSubLive"] < 0 and new_df.loc[i, "NumSubLive"] < new_df.loc[i - 1, "NumSubLive"]):
                    f.write("{0},{1},{2},{3},{4},{5}\n".format(os.path.basename(path).split("_timeline")[0], i, 
                    new_df.loc[i, "NumSubLive"], count_after_10s, count_after_20s, count_after_30s))


        # 新たなタイムラインを結合
        new_df.insert(0, 'Filename', os.path.basename(path))        
        if first_df:
            df = new_df
            first_df = False
        else:
            df = df.append(new_df, ignore_index=True)

        # チーム合計キル数の集計
        total_our_kill = 0
        total_opponent_kill = 0
        for i in range(1, len(new_df)):
            if new_df.loc[i - 1, "OpponentNumLive"] > new_df.loc[i, "OpponentNumLive"]:
                total_our_kill += new_df.loc[i - 1, "OpponentNumLive"] - new_df.loc[i, "OpponentNumLive"]
            if new_df.loc[i - 1, "OurNumLive"] > new_df.loc[i, "OurNumLive"]:
                total_opponent_kill += new_df.loc[i - 1, "OurNumLive"] - new_df.loc[i, "OurNumLive"]

        # statsをcsvで出力
        our_kill_per_min = total_our_kill / len(new_df) * 60
        opponent_kill_per_min = total_opponent_kill / len(new_df) * 60
        df_stats = df_stats.append(pd.Series([
            os.path.basename(path).split("_timeline")[0],
            (1 if win else (-1 if lose else 0)), len(new_df),
            total_our_kill, total_opponent_kill, our_kill_per_min, opponent_kill_per_min, (our_kill_per_min if win else (opponent_kill_per_min if lose else 0)),
            (our_kill_per_min - opponent_kill_per_min if win else (opponent_kill_per_min - our_kill_per_min if lose else 0)),
            win_team_without_penalty, game_time_without_penalty, (1 if (1 if win else (-1 if lose else 0)) != win_team_without_penalty else 0)
            ], index=df_stats.columns), ignore_index=True)

        num_input_csv += 1

    # 試合ごとの詳細情報をcsv出力
    df_stats.to_csv(os.path.join(output_dir, filename_head + "_stats.csv"), encoding="utf_8_sig")

    # タイムラインを結合したものを出力
    df.to_csv(os.path.join(output_dir, filename_head + "_timeline_union.csv"), encoding="utf_8_sig")

    # まとめのcsvを出力
    with open(os.path.join(output_dir, filename_head + "_summary.csv"), mode='w') as f:
        f.write("試合数,{0}\n".format(num_game))
        f.write("勝敗数,{0}\n".format(num_win + num_lose))
        f.write("勝利数/勝率,{0},{1:0.1f}%\n".format(num_win, num_win / (num_win + num_lose) * 100))

        f.write("カウント先制数/先制率,{0},{1:0.1f}%\n".format(num_first_initiative, num_first_initiative / num_game * 100))
        f.write("カウント先制時の勝利数/勝率,{0},{1:0.1f}%\n".format(num_first_initiative_and_win, num_first_initiative_and_win / num_first_initiative * 100))
        f.write("カウント先制取られ時の勝利数/勝率,{0},{1:0.1f}%\n".format(num_lost_first_initiative_and_win, num_lost_first_initiative_and_win / (num_win + num_lose - num_first_initiative) * 100))
        f.write("カウント先制チームの勝率,,{0:0.1f}%\n".format((num_first_initiative_and_win + num_lost_first_initiative_and_lose) / (num_win + num_lose) * 100))

        f.write("先制キル数/先制率,{0},{1:0.1f}%\n".format(num_first_kill, num_first_kill / num_game * 100))
        f.write("先制キル時の勝利数/勝率,{0},{1:0.1f}%\n".format(num_first_kill_and_win, num_first_kill_and_win / num_first_kill * 100))
        f.write("先制キル取られ時の勝利数/勝率,{0},{1:0.1f}%\n".format(num_lost_first_kill_and_win, num_lost_first_kill_and_win / (num_win + num_lose - num_first_kill) * 100))
        f.write("先制キル取得チームの勝率,,{0:0.1f}%\n".format((num_first_kill_and_win + num_lost_first_kill_and_lose) / (num_win + num_lose) * 100))

        f.write("人数差+2先制数/先制率,{0},{1:0.1f}%\n".format(num_first_2sub, num_first_2sub / num_game * 100))
        f.write("人数差+2先制時の勝利数/勝率,{0},{1:0.1f}%\n".format(num_first_2sub_and_win, num_first_2sub_and_win / num_first_2sub * 100))
        f.write("人数差+2先制取られ時の勝利数/勝率,{0},{1:0.1f}%\n".format(num_lost_first_2sub_and_win, num_lost_first_2sub_and_win / num_lost_first_2sub * 100))
        f.write("人数差+2先制チームの勝率,,{0:0.1f}%\n".format((num_first_2sub_and_win + num_lost_first_2sub_and_lose) / (num_win + num_lose) * 100))


if __name__ == "__main__":
    main(default_input_dir, default_output_dir)
