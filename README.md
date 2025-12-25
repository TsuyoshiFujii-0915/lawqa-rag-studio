# LawQA-RAG-Studio

日本の法令コーパス（e-Gov）＋ lawqa_jp を使い、RAG構成を `config.yaml` だけで切り替えて簡単に性能評価を実行できるフレームワークです。  
eval（評価）を軸に、同じ設定で serve（API + フロント）も利用できます。

## 主な機能

- eval：lawqa_jp を使った評価（Accuracy / Macro F1 / 時間 / トークン）
- dense / sparse / hybrid / rerank などの機能選択・切り替え
- Qdrant の local / server / in-memory 対応
- 評価成果物の保存（metrics.json / details.csv / config_snapshot.yaml）
- serve：FastAPI + シンプルなチャットUI（評価結果の確認や対話用）

## 必要要件

- Python 3.11+
- Docker または uv
- Node.js / npm（フロントをローカルで動かす場合）

## 環境変数

`.env` を用意して設定します（例は `.env.example`）。

- `OPENAI_API_KEY`：OpenAI API を使う場合
- `LMSTUDIO_API_KEY`：LM Studio を使う場合（ダミーでも可）
- `LAWQA_CORS_ORIGINS`：フロントからのアクセス許可（必要なら）
- `VITE_API_BASE`：フロントの API 接続先（必要なら）

## クイックスタート

### eval

- Docker
```bash
make eval
```

- uv
```bash
make eval-uv
```

### serve（API + フロント）

- Docker
```bash
make serve
```

- uv
```bash
make serve-uv
```

フロントは `http://localhost:5173`、API は `http://localhost:8000` で起動します。

### CONFIG オプション

`CONFIG=...` で任意の config を指定できます。（省略でデフォルトの `config/config.yaml` が選択されます）

```bash
make eval-uv CONFIG=config/config.gpt-oss.yaml
make serve   CONFIG=config/config.gpt-oss.docker.yaml
```

### recreate オプション

`recreate` を付けると、既存の Qdrant コレクションを削除して再作成します。
インデックスを作り直したい場合に使用してください。

```bash
make eval-recreate CONFIG=config/config.yaml
make serve-recreate CONFIG=config/config.yaml
```

### Docker x LM Studio の注意

Docker からホストの LM Studio に接続する場合は、`config` の `llm.lmstudio.base_url` を以下にします。

```
http://host.docker.internal:1234/v1
```

ローカル（uv）では `http://localhost:1234/v1` を使います。

## 評価

- lawqa_jp の 4択問題に対する **正解率（Accuracy）** と **Macro F1** を算出します
- 各クエリの **応答時間** と **トークン消費量** を記録します
- 結果は `outputs/<experiment_name>/metrics.json` に出力されます

例:

```json
{
  "accuracy": 0.75,
  "macro_f1": 0.73,
  "num_examples": 100,
  "pdf_only_skipped": 10,
  "timing": {
    "total_duration_sec": 1234.5,
    "avg_response_time_sec": 2.3,
    "min_response_time_sec": 0.8,
    "max_response_time_sec": 5.2
  },
  "token_usage": {
    "total_prompt_tokens": 150000,
    "total_completion_tokens": 25000,
    "total_tokens": 175000
  }
}
```

### 評価成果物の使い方

- `metrics.json`：集計指標（Accuracy / Macro F1 / 時間 / トークン）  
  主要な性能比較は `metrics.json` の内容で把握できます。
- `details.csv`：問題ごとの予測結果・正解ラベル・応答時間など  
  誤答分析や失敗例の抽出に使います。
- `config_snapshot.yaml`：評価実行時の設定スナップショット  
  再現性確保と結果比較のベースに使います。

### 評価の注意事項

- lawqa_jp には e-Gov 以外の PDF 資料を参照する問題がありますが、本プロジェクトでは e-Gov の XML のみ取得します
- そのため PDF-only の問題は自動で除外されます
- `eval.max_examples` に数値を指定すると、PDF-only 除外後の問題リストの先頭からその件数だけを評価し、`null` の場合は e-Gov を参照するすべての問題（130件）を使用して評価します

## フロントエンド

- Vite + React
- AI回答は Markdown 表示に対応（見出し / 表 / コードブロック / リンク）
- チャンク一覧をポップアップで表示可能

ローカルでフロントだけ起動する場合：

```bash
cd frontend
VITE_API_BASE=http://localhost:8000 npm run dev
```

## データの取得とインデックス作成

e-Gov XML の取得とインデックス作成は以下で行います。
lawqa_jp の参照先には PDF も含まれますが、本プロジェクトでは e-Gov の XML のみ取得対象としています。

```bash
uv run lawqa-rag-studio fetch_egov --dest data/egov/xml
uv run lawqa-rag-studio ingest --config config/config.yaml
```

## Make タスク一覧

```bash
make eval
make eval-recreate
make serve
make serve-recreate
make eval-uv
make eval-recreate-uv
make serve-uv
make serve-recreate-uv
make up
make down
make build
make gendocs
make gendocs-uv
make test
make test-uv
```

---

必要に応じて `config` を複数用意して切り替えてください。  
細かい構成やパラメータは `config/` を参照。
