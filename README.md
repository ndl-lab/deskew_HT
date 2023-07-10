# NDLOCR用資料画像の傾き補正モジュール(NDLOCR ver.1.0用)

画像の傾きを補正するモジュールのリポジトリです。

本プログラムは、国立国会図書館が株式会社モルフォAIソリューションズに委託して作成したものです。

本プログラム内の[alyn3](alyn3)は以下のリポジトリのコードを参照し、python3化・高速化等を行い作成しました。

[kakul/Alyn/alyn](https://github.com/kakul/Alyn)

本プログラムの新規開発部分は、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については
[LICENSE](./LICENSE
)をご覧ください。

# 概要

入力画像の直線要素を検出することで画像の傾きを推定し、傾きを補正した画像を出力する。

出力画像は元画像の領域が欠損しないように拡大して保存する仕様となっており、
この際に元画像領域外の部分は指定した濃さのグレー(デフォルトは白)で塗りつぶされる。

推定した傾きの数値情報は、オプション（後述）を指定することでテキストファイルとして出力できる。


# 使い方

指定パスの入力画像または指定ディレクトリ内の画像の傾きを推定し補正する。

※補正角度は±45度以内に限る。

```
python3 run_deskew.py INPUT [-o OUTPUT] [-s SKEW_MAX] [-a ANGLE_ACC] [-m METHOD]
```

positional arguments:
```
  input                 入力画像のパス、または入力画像を格納したディレクトリのパス
```

optional arguments:
```
  -h, --help            ヘルプメッセージを表示して終了
  -o OUT, --out OUT     出力ファイルのパス(INPUTが画像ファイルの時、default: out.jpg)または
                        出力ディレクトリのパス(INPUTがディレクトリの時、default: out)
  -l LOG, --log LOG     推定した傾きを保存するテキストファイルのパス。指定なしの場合出力されない
                        処理画像一枚ごとに次の形式で指定ファイルの最終行に追加する。
                        output format:
                        Image_file_path <tab> Estimated_skew_angle[deg]
  -s SKEW_MAX, --skew_max SKEW_MAX
                        推定する傾きの最大角度[deg] default: 4.0[deg]
                        0より大きい45以下の値を指定する。大きくするほど処理時間は増加
  -a ANGLE_ACC, --angle_acc ANGLE_ACC
                        傾きの探索を何度単位で行うか。default: 0.5[deg]
                        0より大きいSKEW_MAX以下の値を指定する。小さくするほど処理時間は増加。
  -rw ROI_WIDTH, --roi_width ROI_WIDTH
                        直線検出の対象とする関心領域の画像全体に対する水平方向の割合
                        0.0より大きい1.0以下の数 default: 1.0(水平方向全体)
  -rh ROI_HEIGHT, --roi_height ROI_HEIGHT
                        直線検出の対象とする関心領域の画像全体に対する鉛直方向の割合
                        0.0より大きい1.0以下の数 default: 1.0(鉛直方向全体)
  -m METHOD, --method METHOD
                        画像回転時の補完手法。以下の整数値で指定する。
                        0: Nearest-neighbor  1: Bi-linear(default)
                        2: Bi-quadratic      3: Bi-cubic
                        4: Bi-quartic        5: Bi-quintic
  -g GRAY, --gray GRAY  出力画像において、元画像領域の外側を補完するグレーの濃さ
                        0(黒) 以上 1.0(白)以下で指定する。default: 1.0(白)
  -q QUALITY, --quality QUALITY
                        Jpeg画像出力時の画質。
                        1が最低画質で最小ファイルサイズ、100が最高画質で最大ファイルサイズ。
                        [1, 100], default: 100
  --short SHORT         出力画像の短辺の長さ。アスペクト比は維持したままリサイズする。
                        指定しなかった場合オリジナルサイズで出力される。
  -v, --version         プログラムのバージョンを表示して終了
```
