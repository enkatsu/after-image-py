# after-image-py

森山祥太郎, 遠藤勝也

Processing 版 [Afterimage](../Afterimage) の Python 移植・拡張。
カメラに映った人物を YOLO + ByteTrack で追跡し、各人物が一定距離以上動いたタイミングでフレーム全体を記録、過去のスナップショットを多重露光や線画として現在フレームに重ねて残像表現を生成する。
同時に最大 5 人まで保持し、6 人目が記録された時点で最も古く登場した人物の残像をバッファから破棄する。

## ファイル構成

```
after-image-py/
├── pyproject.toml             # パッケージ定義
├── requirements.txt
├── README.md
├── .env / .env.example        # 環境変数による実行時設定
├── models/
│   └── yolov8n.pt             # 初回起動時に自動ダウンロード
└── src/
    └── after_image/
        ├── __init__.py
        ├── __main__.py        # エントリ。`python -m after_image` で実行
        ├── tracker.py         # Track / TrackManager / SHOULD_RECORD
        ├── effects.py         # RENDERERS と各実装、線画化ヘルパ
        └── debug.py           # annotate / render_buffer_grid / list_cameras
```

## 動作

1. カメラフレームを取得し、YOLOv8 の `model.track(persist=True)` で人物クラスのみを検出・ID 付け（ByteTrack）。
2. 検出された各人物について bbox 中心を計算し、前回記録時の中心から閾値以上動いていれば（または初検出なら）現在のフレーム全体を `(frame_count, frame)` のタプルとしてその人物のバッファに追加。
3. 各人物のバッファは `BUFFER_LENGTH` の固定長 deque。新しいフレームが入ると最古のフレームが押し出される。
4. 新規 ID の登録時に総人数が `MAX_PEOPLE` を超えたら、最も古く登録された人物（`people[0]`）を丸ごと破棄。
5. 全人物のバッファを統合して `frame_count` で重複排除し、選択中のエフェクト関数で合成して描画。

## データ構造

```
TrackManager
└── people: list[Track]                              # 登場順（古い → 新しい）
     └── Track
          ├── id: int                                # ByteTrack の追跡 ID
          ├── last_center: (cx, cy)                  # 前回記録時の bbox 中心
          └── frames: deque[(frame_count, ndarray)]  # 最新 BUFFER_LENGTH 枚
```

## エフェクト

`EFFECT` 環境変数で実行時に切替可能。

| キー | 関数 | 概要 |
| --- | --- | --- |
| `multi` | `multiple_exposure` | 多重露光。全レイヤーを半透明で積層しフィルム調 |
| `sketch` | `sketch_overlay` | 鉛筆画調。Sobel 線画を MULTIPLY で階調積層 |
| `plotter` | `plotter_overlay` | プロッター調。Canny で 1 px 二値線画にして積層（既定） |

## 記録条件

`TRIGGER` 環境変数で実行時に切替可能。

| キー | 取得元 | 概要 |
| --- | --- | --- |
| `distance` | `make_should_record_by_distance()` | 前回記録時から閾値（既定 80 px）以上動いたら記録（既定） |
| `always` | `should_record_always` | 検出フレームすべてを記録 |

`should_record(track, cx, cy) -> bool` のシグネチャを満たせば独自関数を `tracker.SHOULD_RECORD` に追加できる。

## デバッグモード

`DEBUG=1` を `.env` またはシェルで設定すると以下が有効になる。`1` / `true` / `yes` のみが真値として扱われ、`0` や空は偽。

- メインウィンドウ上の bbox + ID + バッファ枚数表示 (`annotate`)
- バッファ一覧の別ウィンドウ表示 (`render_buffer_grid`)
- 起動時のカメラインデックス利用可否の列挙 (`list_cameras`)

## パラメータ

| 名前 | 既定値 | 場所 | 説明 |
| --- | --- | --- | --- |
| `PERSON_CLASS_ID` | `0` | `__main__.py` | YOLO の人物クラス ID（COCO の `person`） |
| `CAMERA_INDEX` | `1` | `__main__.py` | `cv2.VideoCapture` のデバイスインデックス |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `1280` / `720` | `__main__.py` | キャプチャ解像度 |
| `MODEL_PATH` | `<PROJECT_ROOT>/models/yolov8n.pt` | `__main__.py` | YOLO モデルファイルの絶対パス |
| `BUFFER_LENGTH` | `5` | `tracker.py` | 1 人あたりに保持する残像フレーム数 |
| `MAX_PEOPLE` | `5` | `tracker.py` | 同時追跡できる人物数の上限 |
| `threshold` | `80` | `make_should_record_by_distance` 引数 | 距離トリガの閾値（ピクセル） |
| `current_alpha` | `0.1` | `multiple_exposure` 引数 | 現在フレームを残像の上に重ねる際の不透明度 |
| `thumb_width` / `thumb_height` | `240` / `135` | `render_buffer_grid` 引数 | デバッグサムネイルのサイズ |

## セットアップ

Python 3.13 を想定。

```sh
python3.13 -m venv .venv
.venv/bin/pip install -e .       # editable install で `after_image` パッケージを登録
cp .env.example .env             # 必要に応じて編集
```

初回起動時に `models/yolov8n.pt`（約 6 MB）が自動ダウンロードされる。
`requirements.txt` も用意してあるが、依存定義は `pyproject.toml` 側が単一ソース。

## 実行

```sh
.venv/bin/python -m after_image
```

`q` または `Esc` で終了。

環境変数で動作を切替：

```sh
EFFECT=sketch DEBUG=1 .venv/bin/python -m after_image
EFFECT=multi TRIGGER=always .venv/bin/python -m after_image
```

## 参考

- [../Afterimage](../Afterimage) — Processing による元実装
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) — YOLO + ByteTrack
- [theskumar/python-dotenv](https://github.com/theskumar/python-dotenv) — `.env` サポート
