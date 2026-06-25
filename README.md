# after-image-py

森山祥太郎, 遠藤勝也

Processing 版 [Afterimage](../Afterimage) の Python 移植・拡張。
カメラに映った人物を YOLO + ByteTrack で検出し、3 種類の表現で残像を作る。

## モード

| モード | コマンド | 入力単位 | 表現 |
| --- | --- | --- | --- |
| `snapshot` | `python -m after_image.snapshot` | 1 人につき 5 枚の静止画 | 各人物の過去フレームを毎フレーム合成。距離トリガで記録 |
| `datamosh` | `python -m after_image.datamosh` | シーン単位の mp4 を最大 5 本 | クリップを結合してデータモッシュ（I-frame 除去 / P-frame 複製） |
| `multiclip` | `python -m after_image.multiclip` | シーン単位の mp4 を最大 5 本 | 5 本を並列ループ再生して多重露光合成（多人数の残像が同時に画面に存在） |

`snapshot` は人物単位、`datamosh` と `multiclip` はシーン単位で素材を蓄積する。
`datamosh` と `multiclip` は同じ録画機構（`SceneRecorder`）を共有するため、`clips/` ディレクトリの mp4 を相互に流用できる。

## ファイル構成

```
after-image-py/
├── pyproject.toml
├── requirements.txt
├── .env / .env.example
├── README.md
├── LICENSES/
│   └── datamoshing-UNLICENSE.txt   # mosh.py 由来のアトリビューション
├── models/
│   ├── yolov8n.pt                  # 初回起動時に自動ダウンロード（torch バックエンド）
│   └── yolov8n_ncnn_model/         # NCNN エクスポート済みモデル（既定で優先利用）
├── clips/                          # datamosh/multiclip が録画したシーン mp4（FIFO）
├── moshed/                         # datamosh が生成したデータモッシュ済 mp4（FIFO）
└── src/after_image/
    ├── recorder.py                 # 【共通】SceneRecorder
    ├── tracker.py                  # 【共通】Track / TrackManager / SHOULD_RECORD
    ├── effects.py                  # 【共通】RENDERERS (multi / sketch / plotter)
    ├── snapshot/                   # 静止画スナップショット合成版
    │   ├── __main__.py
    │   └── debug.py
    ├── datamosh/                   # データモッシュ版
    │   ├── __main__.py
    │   ├── mosh.py                 # I-frame 除去 / P-frame 複製の実装
    │   ├── pipeline.py             # concat + mosh ラッパー
    │   ├── player.py               # 完成 mp4 のループ再生
    │   └── worker.py               # 別スレッドで mosh 生成（latest-wins）
    └── multiclip/                  # 多重露光動画版
        ├── __main__.py
        └── player.py               # 並列再生 + shim track 生成
```

## snapshot モード

1. カメラフレームを取得し、YOLOv8 の `model.track(persist=True)` で人物のみ検出・ID 付け。
2. 各人物について bbox 中心が前回記録時から閾値以上動いていればフレーム全体を `(frame_count, frame)` でバッファに追加。
3. 各人物のバッファは `BUFFER_LENGTH` 長の固定 deque。
4. 新規 ID 登録時に総人数が `MAX_PEOPLE` を超えたら、最古の人物を破棄。
5. 全人物のバッファを統合して `EFFECT` で合成し描画。

エフェクト切替（`EFFECT` 環境変数）：

| キー | 関数 | 概要 |
| --- | --- | --- |
| `multi` | `multiple_exposure` | 多重露光。全レイヤーを半透明積層しフィルム調 |
| `sketch` | `sketch_overlay` | 鉛筆画調。Sobel 線画を MULTIPLY で階調積層 |
| `plotter` | `plotter_overlay` | プロッター調。Canny で 1 px 二値線画積層（既定） |

記録条件切替（`TRIGGER` 環境変数）：

| キー | 概要 |
| --- | --- |
| `distance` | 前回記録時から閾値（既定 80 px）以上動いたら記録（既定） |
| `always` | 検出フレームすべてを記録 |

`DEBUG=1` で bbox / バッファサムネイル / カメラ列挙が有効。

## datamosh モード

シーン単位で mp4 を録画してストックし、データモッシュ済み mp4 を生成・ループ再生する。

### 状態遷移

```
LIVE          ──[初回 mosh 完成]──▶  MOSHED_PLAYING
MOSHED_PLAYING ──[新 mosh 完成]──▶  MOSHED_PLAYING（次ループ頭で差し替え）
```

- 起動直後はクリップが無いのでライブビュー表示
- 録画は常時稼働（再生状態に関係なく）
- 一度 MOSHED に入ったらライブには戻らない

### 録画

`recorder.SceneRecorder` が以下を満たすシーンを mp4 として書き出す：

- 検出が始まったらフレームを記録開始
- 検出ゼロが `SCENE_END_SILENCE_FRAMES` 連続したらシーン終了
- 1 シーンが `SCENE_MAX_SECONDS` 秒を超えたら強制終了して次のシーンへ
- `SCENE_MIN_FRAMES` 未満のシーンは破棄
- `MAX_STOCK` 本を超えたら最古のクリップを物理削除（FIFO）

### 生成パイプライン

`MOSH_DELTA` で 2 つのモードを切替：

- **`MOSH_DELTA=0`（I-frame 除去 / 既定）**：全クリップを `ffmpeg concat` で結合し、2 本目以降の I-frame を一括削除。シーン境目で前クリップの絵が後クリップの動きベクトルで歪んで溶ける。
- **`MOSH_DELTA=N`（P-frame 複製 / N > 0）**：各クリップを個別に N フレーム delta mosh してから結合。各シーン冒頭の N フレーム分の動きがそのシーン内で繰り返し再生される（動きの軌跡が滲み続ける）。

mosh.py の本体は [tiberiuiancu/datamoshing](https://github.com/tiberiuiancu/datamoshing) のフォーク。詳細は [`LICENSES/datamoshing-UNLICENSE.txt`](LICENSES/datamoshing-UNLICENSE.txt) を参照。

### バックグラウンド生成

`worker.MoshWorker` がデーモンスレッドで `build_moshed()` を実行する。
生成中に新規 request が来た場合は **latest-wins** で最新のストックだけ覚えておき、終わり次第続けて処理する。完成 mp4 のパスは `MoshedPlayer.update_source()` を経由して次のループ頭で差し替え。`moshed/` には最新 `keep_recent` 本（既定 3）だけ残し、古いものは worker が削除する。

### 手動生成（CLI）

```sh
# 既存の clips/*.mp4 で moshed/moshed.mp4 を生成（録画ループは走らない）
.venv/bin/python -m after_image.datamosh.pipeline
MOSH_DELTA=5 .venv/bin/python -m after_image.datamosh.pipeline
```

## multiclip モード

ストックされている mp4 を **すべて並列にループ再生** し、毎フレーム各クリップから 1 フレームずつ取り出して既存の `RENDERERS` で合成する。datamosh と違って絵が複数枚同時に画面に存在するので、**複数人の残像が同時に動く**表現になる。

エフェクトは `EFFECT=multi`（既定） / `sketch` / `plotter`。snapshot と同じ `RENDERERS` を流用しているため、見た目の傾向もそれに準ずる。

各クリップは異なる長さでループするので、再生位相が次第にズレて新鮮味が出る。

## 共通の仕組み

### 人物検出

3 モードとも YOLOv8 + ByteTrack を使用。`PERSON_CLASS_ID=0`（COCO の `person`）のみ検出対象にしている。

### 環境変数

`.env` または `.env.example` を参照。主なもの：

| 名前 | 既定値 | 用途 |
| --- | --- | --- |
| `CAMERA_INDEX` | `1` | `cv2.VideoCapture` のデバイス番号 |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `1280` / `720` | キャプチャ解像度 |
| `MODEL_PATH` | `models/yolov8n_ncnn_model`（無ければ `models/yolov8n.pt`） | YOLO モデルのパス。NCNN モデルのディレクトリがあれば自動で優先 |
| `DEBUG` | 空 | `1` / `true` / `yes` でデバッグ表示有効 |
| `EFFECT` | `plotter` (snapshot) / `multi` (multiclip) | 合成エフェクト |
| `TRIGGER` | `distance` | snapshot の記録条件 |
| `BUFFER_LENGTH` | `5` | snapshot の 1 人あたり残像枚数 |
| `MAX_PEOPLE` | `5` | snapshot の同時追跡人数上限 |
| `CLIPS_DIR` | `clips` | datamosh/multiclip のクリップ出力先 |
| `MOSHED_DIR` | `moshed` | datamosh の moshed mp4 出力先 |
| `MAX_STOCK` | `5` | datamosh/multiclip のストック上限 |
| `SCENE_MAX_SECONDS` | `5.0` | 1 シーンの最大長 |
| `SCENE_END_SILENCE_FRAMES` | `15` | 何フレーム無検出でシーン終了とみなすか |
| `SCENE_MIN_FRAMES` | `30` | これ未満のシーンは破棄 |
| `MOSHED_FPS` | `30` | datamosh 出力 fps |
| `MOSH_DELTA` | `0` | 0 = I-frame 除去、>0 = P-frame 複製 |

## セットアップ

Python 3.13 を想定。`ffmpeg` / `ffprobe` が PATH に通っている必要がある（datamosh モードのみ）。

```sh
python3.13 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env             # 必要に応じて編集
```

初回起動時に `models/yolov8n.pt`（約 6 MB）が自動ダウンロードされる。

### 推論バックエンド（NCNN / torch）

既定では `models/yolov8n_ncnn_model/` があればそれを優先的にロードし、**NCNN** で推論する。
torch の CPU カーネルを通さないため、PyPI の torch ホイールが原因で発生する
`Illegal instruction (core dumped)` を回避できる（Raspberry Pi など aarch64 ボードで頻発）。
NCNN モデルが無い場合は `models/yolov8n.pt` を **torch** バックエンドでロードする。

NCNN モデルはリポジトリに同梱済み。自分で再生成する場合は、torch が正常に動く
マシン（Mac など）で次を実行する（生成物の `.param` / `.bin` は移植可能なので、
そのまま Pi へコピーして使える）:

```sh
.venv/bin/python -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt').export(format='ncnn')"
# 生成された yolov8n_ncnn_model/ を models/ 配下へ配置
```

> 補足: ultralytics の NCNN エクスポートには `pnnx` が必要。`pnnx` の同梱バイナリは
> ビルド時の OS より新しい macOS では動くが、**古い macOS では起動しない**ことがある
> （`built for macOS X which is newer than running OS`）。その場合は
> `pip install "pnnx==20240819"` のように、稼働 OS に合う古いバージョンへ下げる。

## 実行

```sh
# snapshot（静止画スナップショット合成）
.venv/bin/python -m after_image.snapshot
EFFECT=sketch DEBUG=1 .venv/bin/python -m after_image.snapshot

# datamosh（I-frame 除去）
.venv/bin/python -m after_image.datamosh
MOSH_DELTA=5 .venv/bin/python -m after_image.datamosh   # P-frame 複製

# multiclip（多重露光動画版）
.venv/bin/python -m after_image.multiclip
EFFECT=plotter .venv/bin/python -m after_image.multiclip
```

いずれも `q` または `Esc` で終了。

## 参考

- [../Afterimage](../Afterimage) — Processing による元実装
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) — YOLO + ByteTrack
- [tiberiuiancu/datamoshing](https://github.com/tiberiuiancu/datamoshing) — `mosh.py` のフォーク元（Unlicense）
- [theskumar/python-dotenv](https://github.com/theskumar/python-dotenv) — `.env` サポート
