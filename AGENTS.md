# LawQA-RAG-Studio

---

## 1. プロジェクト概要

### 1.1 ゴール

* **日本の法令コーパス（e-Gov）＋ lawqa_jp** を使い、
* さまざまな **RAG構成（チャンク・埋め込み・検索・リランキング・LLM）** を
* **config.yaml だけで切り替えてベンチマーク／対話利用できる** フレームワークを提供する。

### 1.2 モード

1. **eval モード**

   * 指定された `config.yaml` に基づき RAG パイプラインを構築
   * lawqa_jp の問題セットを解かせて評価指標を算出

2. **serve モード**

   * 指定された `config.yaml` に基づき RAG パイプラインを構築
   * シンプルな Web API + チャットUIで対話利用できるようにする

---

## 2. 全体アーキテクチャ

テキストでのざっくり図：

1. **データレイヤー**

   * e-Gov XML → 法令ツリー（構造付き中間表現） → チャンク生成 → Qdrant に格納
   * lawqa_jp → 内部の問題表現へロード

2. **RAG パイプライン**

   * Query 前処理（必要なら：HyDE / マルチクエリなど）
   * Dense / Sparse / Hybrid 検索（Qdrant）([qdrant.tech][1])
   * Optional リランキング（bge-reranker-v2-m3 系）([Hugging Face][2])
   * コンテキスト構成
   * LLM 呼び出し（Responses API / LM Studio OpenAI互換 API）([OpenAI Platform][3])

3. **モード別**

   * eval: lawqa_jp の問題を流してスコア計算
   * serve: HTTP API + フロントエンドUI

---

## 3. 技術スタック仕様

### 3.1 言語・ランタイム

* Python **3.11+**
* 型ヒント必須（`mypy` 通るレベルを目標だが、最初は optional）

### 3.2 主要ライブラリ

* LLM クライアント：`openai`（Python公式 SDK）([OpenAI Platform][3])
* ベクトル DB：`qdrant-client`（dense + sparse named vectors 利用）([qdrant.tech][1])
* 埋め込み：

  * Dense: OpenAI `text-embedding-3-small` / `text-embedding-3-large`([OpenAI Platform][4])
  * Sparse: `bizreach-inc/light-splade-japanese-56M`（日本語特化の軽量SPLADEモデル）([Hugging Face][5])
* Reranker：

  * `BAAI/bge-reranker-v2-m3` もしくは `hotchpotch/japanese-bge-reranker-v2-m3-v1`（日本語用）([Hugging Face][2])
* Web API: `FastAPI` + `uvicorn`
* Frontend: 最初はシンプルな HTML/JS（必要なら後で React に拡張）
* パッケージ / スクリプト実行：`uv`
* タスクランナー：`Makefile`（`uv` と `docker compose` コマンドを集約）

### 3.3 モデル

* **LLM（RAG本体）**（Responses API 経由）([OpenAI Platform][3])

  * `gpt-5.1`
  * `gpt-5`
  * `gpt-5-mini`
  * `gpt-5-nano`
  * `gpt-oss:20b`（LM Studio 経由・OpenAI互換 HTTP）([MOT NOTE][6])

* **Embeddings API**([OpenAI Platform][7])

  * `text-embedding-3-small`
  * `text-embedding-3-large`

---

## 4. ディレクトリ構成（提案）

```text
lawqa-rag-studio/
  pyproject.toml
  uv.lock
  compose.yaml
  Dockerfile
  Makefile

  config/
    config.example.yaml

  src/
    lawqa_rag_studio/
      __init__.py
      cli.py             # エントリポイント (eval / serve / ingest)

      config/
        schema.py        # pydantic などで config 構造定義
        loader.py        # YAML ロード + バリデーション

      data/
        egov_downloader.py  # e-Gov XML ダウンロード
        egov_parser.py      # XML -> 法令ツリー
        law_tree.py         # LawNode 定義
        markdown_exporter.py # 必要なら Markdown への変換
        lawqa_loader.py     # lawqa_jp ローダー

      chunking/
        base.py
        fixed.py
        hierarchy.py

      embeddings/
        dense/
          openai.py
        sparse/
          splade.py

      vectorstore/
        qdrant_client.py   # Qdrant の抽象ラッパー

      retrieval/
        query_models.py    # HyDE / multi-query など
        dense.py
        sparse.py
        hybrid.py          # RRF / Linear
        rerank.py
        pipeline.py        # 全体の Retrieve パイプライン

      llm/
        base.py
        openai_responses.py
        lmstudio_openai_compat.py

      rag/
        pipeline.py        # end-to-end RAG (retrieve+generate)

      eval/
        runner.py
        metrics.py
        reporters.py       # JSON/CSV/Markdown レポート

      serve/
        api.py             # FastAPI アプリ
        schemas.py         # API I/O スキーマ

      logging/
        setup.py           # ログ設定
        formatters.py      # カスタムフォーマッター

  tools/
    generate_config_docs.py  # Config選択肢ドキュメント自動生成

  docs/
    config_options.md      # 自動生成されるConfig選択肢一覧

  tests/
    ...
```

---

## 5. config.yaml 仕様

### 5.1 全体スキーマ（例）

```yaml
experiment:
  name: LawQA-RAG-Studio
  seed: 42
  output_dir: outputs

data:
  egov:
    enabled: true
    xml_dir: ./data/egov/xml
    law_tree_cache: ./data/egov/law_tree.jsonl
  lawqa_jp:
    path: ./data/lawqa_jp/selection_randomized.json

vector_store:
  qdrant:
    collection_name: law_ja
    location: local            # local | server | in-memory
    local:
      storage_dir: ./data/qdrant
    server:
      url: http://qdrant:6333   # server モード時のみ使用
      api_key: null              # 環境変数優先（server モード時のみ）

chunking:
  strategy: fixed        # fixed | hierarchy
  fixed:
    max_chars: 1200
    overlap_chars: 200
  hierarchy:
    level: section       # law | chapter | section | article | paragraph
    max_chars_per_chunk: 2000

embedding:
  dense:
    provider: openai
    model: text-embedding-3-small  # or text-embedding-3-large
    batch_size: 32
  sparse:
    enabled: true
    model: bizreach-inc/light-splade-japanese-56M
    batch_size: 16

retriever:
  dense:
    top_k: 50
  sparse:
    top_k: 100
  hybrid:
    combine: rrf         # rrf | linear
    top_k: 20
    linear_weights:
      dense: 0.7
      sparse: 0.3
  rerank:
    enabled: true
    model: BAAI/bge-reranker-v2-m3
    top_k: 5
  hyde:
    enabled: false
    prompt_template: hyde_default
  extra:
    multi_query:
      enabled: false
      num_queries: 3
    context_compression:
      enabled: false
      target_tokens: 1500

llm:
  provider: openai       # openai | lmstudio
  model: gpt-5-mini
  openai:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    rag_mode: plain      # plain | agentic (Responses API専用)
  lmstudio:
    base_url: http://localhost:1234/v1
    api_key_env: LMSTUDIO_API_KEY

eval:
  split: test            # validation | test | all
  max_examples: null     # null で全件
  metrics:
    - accuracy
    - macro_f1
  output_dir: ${experiment.output_dir}/${experiment.name}

serve:
  host: 0.0.0.0
  port: 8000

logging:
  level: INFO             # DEBUG | INFO | WARNING | ERROR
```

### 5.x 運用手順（初回〜更新時）

1. **e-Gov XML ダウンロード（初回のみ）**
   ```bash
   uv run lawqa-rag-studio fetch-egov --dest data/egov/xml \
     --url https://laws.e-gov.go.jp/bulkdownload?file_section=1&only_xml_flag=true
   ```
   * ZIP は展開後に削除され、XML が `data/egov/xml` に展開される。
2. **ベクトルストア構築（必要に応じて再作成）**
   ```bash
   uv run lawqa-rag-studio ingest --config config/config.example.yaml            # 既存があればスキップ
   uv run lawqa-rag-studio ingest --config config/config.example.yaml --recreate # 強制再作成
   ```
3. **評価**
   ```bash
   uv run lawqa-rag-studio eval --config config/config.example.yaml
   uv run lawqa-rag-studio eval --config config/config.example.yaml --recreate-index  # 評価前に再作成
   ```
4. **サーブ**
   ```bash
   uv run lawqa-rag-studio serve --config config/config.example.yaml
   uv run lawqa-rag-studio serve --config config/config.example.yaml --recreate-index # 起動時に再作成
   ```

*Qdrant モード*: `vector_store.qdrant.location` で `local` / `server` / `in-memory` を切替。  
*キー類*: OpenAI/LM Studio を使う場合は環境変数を事前に設定。

Codex には `config/schema.py` でこの YAML を pydantic モデルに落とし込ませる想定。

### 5.2 Config 選択肢一覧

設定可能な選択肢は `src/lawqa_rag_studio/config/constants.py` で一元管理し、ドキュメントは自動生成する。

* **選択肢一覧**: `docs/config_options.md`（自動生成）
* **情報源**: `src/lawqa_rag_studio/config/constants.py`

選択肢を追加・変更した際は、`constants.py` を更新後に `make gendocs` を実行してドキュメントを再生成する。

### 5.3 Config 選択肢ドキュメント自動生成

config の選択肢をコードの定数から自動生成する仕組みを導入する。

#### 5.3.1 定数定義（src/lawqa_rag_studio/config/constants.py）

```python
"""Config選択肢の定数定義。ドキュメント自動生成の源泉となる。"""
from enum import Enum
from typing import NamedTuple


class ConfigOption(NamedTuple):
    """設定選択肢のメタデータ"""
    value: str
    description: str
    default: bool = False


class ChunkingStrategy(Enum):
    FIXED = ConfigOption("fixed", "固定長でのスライディングウィンドウ分割", default=True)
    HIERARCHY = ConfigOption("hierarchy", "法令構造（章・節・条）に基づく分割")


class HierarchyLevel(Enum):
    LAW = ConfigOption("law", "法令全体を1チャンク")
    CHAPTER = ConfigOption("chapter", "章単位で分割")
    SECTION = ConfigOption("section", "節単位で分割", default=True)
    ARTICLE = ConfigOption("article", "条単位で分割")
    PARAGRAPH = ConfigOption("paragraph", "項単位で分割")


class DenseEmbeddingModel(Enum):
    TEXT_EMBEDDING_3_SMALL = ConfigOption(
        "text-embedding-3-small", "OpenAI 小型埋め込みモデル (1536次元)", default=True
    )
    TEXT_EMBEDDING_3_LARGE = ConfigOption(
        "text-embedding-3-large", "OpenAI 大型埋め込みモデル (3072次元)"
    )


class SparseEmbeddingModel(Enum):
    LIGHT_SPLADE_JAPANESE = ConfigOption(
        "bizreach-inc/light-splade-japanese-56M",
        "日本語特化の軽量SPLADEモデル",
        default=True
    )


class HybridCombineMethod(Enum):
    RRF = ConfigOption("rrf", "Reciprocal Rank Fusion", default=True)
    LINEAR = ConfigOption("linear", "正規化スコアの線形結合")


class RerankModel(Enum):
    BGE_RERANKER_V2_M3 = ConfigOption(
        "BAAI/bge-reranker-v2-m3", "多言語対応リランカー", default=True
    )
    JAPANESE_BGE_RERANKER = ConfigOption(
        "hotchpotch/japanese-bge-reranker-v2-m3-v1", "日本語特化リランカー"
    )


class LLMProvider(Enum):
    OPENAI = ConfigOption("openai", "OpenAI API", default=True)
    LMSTUDIO = ConfigOption("lmstudio", "LM Studio経由のOSSモデル")


class LLMModel(Enum):
    GPT_5_1 = ConfigOption("gpt-5.1", "GPT-5.1 最新モデル")
    GPT_5 = ConfigOption("gpt-5", "GPT-5")
    GPT_5_MINI = ConfigOption("gpt-5-mini", "GPT-5 Mini (軽量)", default=True)
    GPT_5_NANO = ConfigOption("gpt-5-nano", "GPT-5 Nano (最軽量)")
    GPT_OSS_20B = ConfigOption("gpt-oss:20b", "OSS 20Bモデル (LM Studio)")


class LogLevel(Enum):
    DEBUG = ConfigOption("DEBUG", "デバッグ情報を含む詳細ログ")
    INFO = ConfigOption("INFO", "通常の実行情報", default=True)
    WARNING = ConfigOption("WARNING", "警告のみ")
    ERROR = ConfigOption("ERROR", "エラーのみ")


class RagMode(Enum):
    PLAIN = ConfigOption("plain", "従来型RAG（retrieve結果をコンテキストとして付加）", default=True)
    AGENTIC = ConfigOption("agentic", "エージェント型RAG（retrieveをtoolとしてLLMが自律呼び出し）")


# 全ての選択肢Enumを登録（ドキュメント生成用）
ALL_CONFIG_ENUMS = {
    "chunking.strategy": ChunkingStrategy,
    "chunking.hierarchy.level": HierarchyLevel,
    "embedding.dense.model": DenseEmbeddingModel,
    "embedding.sparse.model": SparseEmbeddingModel,
    "retriever.hybrid.combine": HybridCombineMethod,
    "retriever.rerank.model": RerankModel,
    "llm.provider": LLMProvider,
    "llm.model": LLMModel,
    "llm.openai.rag_mode": RagMode,
    "logging.level": LogLevel,
}
```

#### 5.3.2 ドキュメント生成スクリプト（tools/generate_config_docs.py）

```python
#!/usr/bin/env python3
"""Config選択肢のドキュメントを自動生成する。

Usage:
    python tools/generate_config_docs.py > docs/config_options.md
    python tools/generate_config_docs.py --format=table  # Markdownテーブル形式
    python tools/generate_config_docs.py --format=json   # JSON形式
"""
import argparse
import json
from lawqa_rag_studio.config.constants import ALL_CONFIG_ENUMS, ConfigOption


def generate_markdown_table() -> str:
    """Markdownテーブル形式でドキュメントを生成"""
    lines = [
        "# Config 選択肢一覧",
        "",
        "> このドキュメントは `scripts/generate_config_docs.py` により自動生成されています。",
        "> 手動で編集しないでください。",
        "",
        "| 設定パス | 選択肢 | 説明 | デフォルト |",
        "|---------|-------|------|-----------|",
    ]
    
    for config_path, enum_cls in ALL_CONFIG_ENUMS.items():
        for member in enum_cls:
            opt: ConfigOption = member.value
            default_mark = "✓" if opt.default else ""
            lines.append(
                f"| `{config_path}` | `{opt.value}` | {opt.description} | {default_mark} |"
            )
    
    return "\n".join(lines)


def generate_json() -> str:
    """JSON形式でドキュメントを生成"""
    result = {}
    for config_path, enum_cls in ALL_CONFIG_ENUMS.items():
        result[config_path] = {
            "options": [
                {
                    "value": member.value.value,
                    "description": member.value.description,
                    "default": member.value.default,
                }
                for member in enum_cls
            ]
        }
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()
    
    if args.format == "table":
        print(generate_markdown_table())
    else:
        print(generate_json())
```

#### 5.3.3 Makefile タスク登録

```makefile
gendocs:
	uv run python tools/generate_config_docs.py > docs/config_options.md
```

#### 5.3.4 運用ルール

* 新しい選択肢を追加する際は、必ず `constants.py` の対応する Enum に追加する
* CI/CD で `gen-config-docs` を実行し、ドキュメントを自動更新
* PR レビュー時に `constants.py` の変更と `docs/config_options.md` の変更が一致しているか確認

---

## 6. データ取得と前処理

### 6.1 e-Gov 法令フルテキスト

#### 6.1.1 取得

* e-Gov 法令検索の **XML一括ダウンロード** を利用。([qdrant.tech][8])
* ダウンロードは手動でも良いが、スクリプト (`egov_downloader.py`) で更新できるようにしておく。
* 利用規約は「政府標準利用規約（第2.0版）」＋ CC BY 4.0 互換相当で、再利用・商用利用可（出典表示必要）。([qdrant.tech][1])

#### 6.1.2 XML → 法令ツリー

`LawNode` 中間表現（src/data/law_tree.py）：

```python
class LawNode(TypedDict):
    node_id: str                  # law_id + path などでユニーク
    type: Literal["law", "chapter", "section",
                  "article", "paragraph", "item"]
    title: str | None             # 条名・章名など
    text: str | None              # そのノード単独の本文
    children: list["LawNode"]
    meta: dict[str, Any]          # law_id, article_num, path, e-gov URL 等
```

* `egov_parser.py` が XML → LawNode ツリーに変換
* ここでは **構造を保持し、テキストはプレーンまたは軽いMarkdown** にする
* 将来のチャンク戦略切り替えのため、XML自体もそのまま保存推奨

### 6.2 Markdown への変換

* 必須ではないが、デバッグや人間確認用に `markdown_exporter.py` を用意
* 例：

  * 法令タイトル：`# 会社法`
  * 章：`## 第一章 総則`
  * 条：`### （第一条 目的）`
  * 項／号：番号をテキストに含める

ただし **RAG のチャンク生成は LawNode ツリーから行う** 前提。

### 6.3 lawqa_jp ローディング

* GitHub の `selection_randomized.json` / `selection_with_reference_randomized.json` を読み込み([OpenAI Platform][3])
* 内部表現：

```python
class LawQaExample(TypedDict):
    id: str
    question: str
    options: list[str]  # ["A...", "B...", ...]
    correct_index: int  # 0-based
    references: list[str]  # e-Gov URL などがあれば
    meta: dict[str, Any]
```

* `eval.cfg.split` = `"validation"` / `"test"` に応じてフィルタリング（元データのフラグに従う）

---

## 7. チャンキング戦略

### 7.1 固定長チャンキング

Config から：

```yaml
chunking:
  strategy: fixed
  fixed:
    max_chars: 1200
    overlap_chars: 200
```

仕様：

* LawNode ツリーを **法令ごとの連結テキスト** に落としてから、
  `max_chars` / `overlap_chars` でスライディングウィンドウ
* 各チャンクには以下メタデータを付与：

  * `law_id`
  * `char_range`（元の法令テキスト内のオフセット）
  * 可能であれば含まれる `article_numbers`（Article ノードから逆算）

### 7.2 ヒエラルキーチャンキング

Config：

```yaml
chunking:
  strategy: hierarchy
  hierarchy:
    level: section          # law | chapter | section | article | paragraph
    max_chars_per_chunk: 2000
```

仕様：

* 指定 `level` で LawNode ツリーを走査し、「そのノード + 子孫ノードのテキスト」を 1 チャンク候補とする
* 候補チャンクが `max_chars_per_chunk` を超える場合：

  * 1つ下の粒度（例：section → article）に分割
* チャンクのメタデータ：

  * `law_id`
  * `hierarchy_path`（["会社法", "第1編", "第1章", "第1節"] 等）
  * `node_type` / `node_id`
  * `article_numbers`（存在すれば）

---

## 8. ベクトルストア（Qdrant）

### 8.1 コレクション設計

* Named vectors を利用：([qdrant.tech][9])

```python
vectors_config = {
  "dense": VectorParams(size=dense_dim, distance="Cosine"),
}
sparse_vectors_config = {
  "sparse": SparseVectorParams(),
}
```

* payload には：

  * `law_id`, `hierarchy_path`, `article_numbers`, `chunking_strategy`, `chunk_id` 等を格納

### 8.2 インデックス

* dense：HNSW（デフォルト）。パフォーマンス調整は config の拡張で対応可。([qdrant.tech][10])
* sparse：Qdrant の sparse vector inverted index を使用。([qdrant.tech][1])

### 8.3 ローカル / サーバー / インメモリ

* `vector_store.qdrant.location` で切り替え：`local` / `server` / `in-memory`
  * local: `vector_store.qdrant.local.storage_dir` に埋め込み Qdrant を保存
  * server: `vector_store.qdrant.server.url` / `api_key` で外部Qdrantに接続
  * in-memory: プロセス内の揮発インスタンス（開発・テスト向け）

---

## 9. 埋め込み

### 9.1 Dense（OpenAI Embeddings API）

* `embedding.dense.model` に `text-embedding-3-small` / `text-embedding-3-large` を指定([OpenAI Platform][4])

* Python SDK 利用例（仕様イメージ）：

  * `client = OpenAI()`
  * `client.embeddings.create(model=..., input=[...])`([OpenAI Platform][7])

* バッチサイズは config の `batch_size`

### 9.2 Sparse（SPLADE）

* Config 例：

```yaml
embedding:
  sparse:
    enabled: true
    model: bizreach-inc/light-splade-japanese-56M
```

* 使用モデル：`bizreach-inc/light-splade-japanese-56M`（日本語特化の軽量SPLADEモデル）

* 実装案：

  * Hugging Face Transformers か、Qdrant 公式の FastEmbed `SparseTextEmbedding` を利用([Hugging Face][5])
  * 出力を `{indices: List[int], values: List[float]}` 形式にして Qdrant の sparse vector に格納([qdrant.tech][1])

---

## 10. Retrieval 戦略

### 10.1 Dense ベクトル検索（基本）

* デフォルトは dense のみ検索。
* `retriever.dense.top_k` で取得チャンク数を制御。
* Qdrant の `query_points` API を利用。([qdrant.tech][11])

### 10.2 Hybrid 検索

* 条件：

  * `embedding.sparse.enabled: true`

* 実装方針：

  1. dense ベクトル検索（top_k）
  2. sparse ベクトル検索（top_k）
  3. スコア統合：

     * **RRF**（Reciprocal Rank Fusion）
     * **Linear**（正規化スコアの線形結合）

* Config：

```yaml
retriever:
  hybrid:
    combine: rrf
    top_k: 20
    linear_weights:
      dense: 0.7
      sparse: 0.3
```

### 10.3 Rerank（bge-m3）

* `retriever.rerank.enabled: true` の場合
* 候補チャンク（例：top_k=20）を reranker にかけて上位 `rerank.top_k` に絞る
* モデルデフォルト：`BAAI/bge-reranker-v2-m3`（多言語） or `hotchpotch/japanese-bge-reranker-v2-m3-v1`（日本語）([Hugging Face][2])

### 10.4 HyDE

* `retriever.hyde.enabled: true` の場合：([OpenAI Platform][3])

フロー：

1. 元クエリ `q` を RAG LLM に投げて「理想的な答え（ドキュメント）」を生成させる
2. その pseudo-doc を dense embedding して検索クエリとして利用
3. 元のクエリも併用するかは config オプション化（v1 は pseudo-doc のみでよい）

### 10.5 その他挟み込み候補（オプション）

Config の `retriever.extra` でオンオフできる形にしておく。実装は後回しでも OK。

1. **Multi-Query**

   * クエリの言い換えを LLM に複数生成させ、それぞれで検索し結果をマージ
2. **Context Compression**

   * 取得した大量チャンクを LLM で要約＋重複除去し、最終コンテキストトークン数を抑える
3. **Diversity Promotion**

   * 同一法令／同一条に偏らないように、ランキング時に多様性ペナルティを入れる

---

## 11. LLM（RAG 本体）

### 11.1 OpenAI Responses API

* Python SDK の `client.responses.create()` を利用([OpenAI Platform][3])

* `llm.provider: openai` の場合：

  * `base_url` / `api_key_env` を config から読み込む
  * `model` は `gpt-5.1` / `gpt-5` / `gpt-5-mini` / `gpt-5-nano`

* プロンプト構成（例）：

  * `system`: 法令QA用の固定プロンプト（configから参照）
  * `messages`: [user(question), context(chunked_text)] を適切なフォーマットで渡す

#### 11.1.1 RAGモード（`openai.rag_mode`）

OpenAI Responses API 使用時のみ、2つのRAGモードを選択可能：

| モード | 説明 |
|--------|------|
| `plain`（デフォルト） | retrieve結果をコンテキストとしてプロンプトに付加する従来型RAG |
| `agentic` | retrieveをtoolとしてResponses APIに渡し、LLMが自律的に呼び出す |

**plain モード（従来型）:**

```
query → retrieve(query) → context構成 → LLM(query + context) → answer
```

**agentic モード:**

```
query → LLM(query, tools=[retrieve]) → (LLMがretrieve呼び出し) → retrieve実行 → LLM(結果) → answer
```

* agentic モードでは、LLMが「検索が必要か」を判断し、必要に応じて複数回の検索や検索クエリの言い換えを自律的に実行可能
* ただし、トークン消費・レイテンシは plain より増加する傾向

※ `lmstudio` プロバイダでは常に plain 相当の動作となる

### 11.2 LM Studio (gpt-oss:20b)

* `llm.provider: lmstudio` の場合：

  * `base_url` を LM Studio の OpenAI互換エンドポイント（例: `http://localhost:1234/v1`） ([MOT NOTE][6])
  * `api_key_env` からトークンを取得（LM Studio はダミーでも可）
  * OpenAI SDK の `base_url` を上書きし、`model: gpt-oss:20b` を指定

※ 実装側には「**OpenAI Responses API 互換で叩けることを前提**」と明記。

---

## 12. RAG パイプライン

`rag/pipeline.py` で統一的なインターフェースを定義：

```python
class RagPipeline(Protocol):
    def answer(self, query: str, *, eval_mode: bool = False) -> RagResult:
        ...
```

`RagResult` には：

```python
class RagResult(TypedDict):
    answer: str                # LLM回答
    used_chunks: list[Chunk]   # 実際にコンテキストに入れたチャンク
    llm_calls: list[dict]      # デバッグ用
    retrieval_info: dict       # dense/sparse scores, rerank, hyde等
```

---

## 13. eval モード仕様

### 13.1 コマンド

```bash
lawqa-rag-studio eval --config config/config.yaml
```

### 13.2 フロー

1. config をロード
2. e-Gov インデックス（Qdrant）が無ければ `ingest` を走らせる（or 明示的に別コマンド要求でも可）
3. lawqa_jp の問題セットをロード
4. 各問題について：

   * `question + options` → RAG に投げる
   * 期待する出力は：

     * **選択肢インデックス（0〜3）** または
     * 選択肢ラベル（"A"/"B"/"C"/"D"）を返すようにプロンプト設計
5. 正解と照合し、メトリクス計算

### 13.3 評価指標

* 必須：

  * **Accuracy**
  * **Macro F1**（クラスごとの偏りチェック用）

### 13.4 出力

* `eval.output_dir` に以下を書き出す：

  * `metrics.json`：集計指標
  * `details.csv`：問題ごとの予測・正解・使用構成
  * `config_snapshot.yaml`：実行時に使用した config のコピー

### 13.5 ログ・計測

#### 13.5.1 時間計測

* 各クエリの回答時間（秒）を記録
* eval 全体の所要時間を記録
* 出力先：`details.csv` に `response_time_sec` カラム追加、`metrics.json` に以下を追加：

```json
{
  "timing": {
    "total_duration_sec": 1234.5,
    "avg_response_time_sec": 2.3,
    "min_response_time_sec": 0.8,
    "max_response_time_sec": 5.2
  }
}
```

#### 13.5.2 トークン数計測

* 各クエリの入力トークン数・出力トークン数を記録
* eval 全体の合計トークン数を記録
* 出力先：`details.csv` に `input_tokens`, `output_tokens` カラム追加、`metrics.json` に以下を追加：

```json
{
  "token_usage": {
    "total_input_tokens": 150000,
    "total_output_tokens": 25000,
    "avg_input_tokens": 300,
    "avg_output_tokens": 50
  }
}
```

#### 13.5.3 詳細ログ出力

* ログレベルは config で設定可能（`DEBUG` / `INFO` / `WARNING` / `ERROR`）
* ログ出力先：

  * コンソール（stderr）
  * ファイル（`eval.output_dir/eval.log`）
* ログフォーマット例：

```
2025-01-15 10:30:45.123 | INFO | eval.runner | Processing query 1/100: "会社法における..."
2025-01-15 10:30:47.456 | DEBUG | retrieval.hybrid | Dense results: 50, Sparse results: 100, Fused: 20
2025-01-15 10:30:48.789 | INFO | eval.runner | Query 1 completed in 3.67s (input: 450 tokens, output: 85 tokens)
```

* Config 追加：

```yaml
logging:
  level: INFO           # DEBUG | INFO | WARNING | ERROR
```

---

## 14. serve モード仕様

### 14.1 コマンド

```bash
lawqa-rag-studio serve --config config/config.yaml
```

### 14.2 API

FastAPI ベースで：

* `GET /health`

* `POST /chat`

  * Request:

    * `message`: str
    * `history`: optional (簡易でいいなら v1 は無視)
  * Response:

    * `answer`: str
    * `chunks`: list[chunk_metadata]
    * `config_hash`: str

* `POST /session/reset`

### 14.3 フロントエンド

* 別で作成するので今はブランクにすること

---

## 15. Docker / uv / タスクランナー

### 15.1 Docker

* `Dockerfile`：

  * Python 3.11 slim ベース
  * `uv` で依存関係をインストール
* `compose.yaml`（※ `docker-compose.yml` ではなく `compose.yaml` を使用）：

  * services:

    * `app`: lawqa-rag-studio
    * `qdrant`: qdrant/qdrant:latest

### 15.2 Makefile（例）

```makefile
up:
	docker compose up -d

down:
	docker compose down

eval:
	docker compose run --rm app lawqa-rag-studio eval --config /app/config/config.yaml

serve:
	docker compose up app

gendocs:
	uv run python tools/generate_config_docs.py > docs/config_options.md

test:
	uv run python -m pytest
```

* `--config` などのオプションはコマンドで上書きできるようにしたい


### 15.3 タスク実行ルール

* 開発系タスクはすべて `make <task>` で実行し、内部で `uv run ...` / `docker compose ...` を呼ぶ。
* 新しいタスクを追加する場合は Makefile にターゲットを追加し、README/ドキュメントを更新する。

---

## 16. テスト

### 16.1 ポリシー

* スキップ対象は **(1) データダウンロード（e-Gov XML 等）** と **(2) 外部API呼び出し（OpenAI / LM Studio 等のネットワークアクセス）** のみ。それ以外は必ず実行可能なテストを用意し、CI でも動かす。
* 乱数は `seed` を固定し、決定的な結果をアサートする。

### 16.2 レイヤ別テスト指針

1. **ユニットテスト（最優先）**
   * `config/loader.py`：必須項目欠落・無効値・型不整合を検証。
   * `chunking/fixed.py`, `chunking/hierarchy.py`：チャンク境界、オーバーラップ、メタデータ（`law_id` / `article_numbers` など）整合性を検証。
   * `data/law_tree.py`：LawNode の親子関係と `node_id` 一意性。
   * `retrieval/hybrid.py`：RRF / linear のスコア融合を固定入力で検証。
   * `retrieval/rerank.py`：入出力整形とトップK切り詰めをモックで検証。
   * `eval/metrics.py`：Accuracy / Macro F1 の境界ケース（全問正解・全問不正解・クラス不均衡）。
   * `logging/formatters.py`：ログフォーマットが設定通りかをキャプチャして確認。

2. **コンポーネント統合テスト**
   * **疑似データ固定**: `tests/fixtures/` に数条文だけの LawNode JSONL とチャンク済みデータを置き、毎回同じ入力で再現性を確保。
   * **Qdrant 依存部**: `qdrant-client` の InMemory (`QdrantClient(location=":memory:")`) のみを使用し、外部プロセスを起動せずに dense / sparse / hybrid 検索とペイロード格納を確認する。
   * **RAG パイプライン**: 上記インメモリ Qdrant を使い、LLM をモックして `rag.pipeline.answer` が決定的な `RagResult` を返すか（使用チャンクID・順位をアサート）。

3. **LLM / Embedding のモック**
   * OpenAI / LM Studio への HTTP は `responses` 等でモックし、決め打ちベクトル・テキストを返す。外部APIへの実呼び出しはスキップ対象とし、テスト失敗扱いにしない。
   * Embedding 出力は固定ベクトルを返し、検索スコア比較を可能にする。

4. **CLI / サービス**
   * `cli.py` の `eval` / `serve` は `--config tests/fixtures/config.min.yaml` を使ってサブプロセス起動し、終了コード0と主要ログ行の有無を確認。
   * `serve/api.py` は FastAPI `TestClient` で `/health` と `/chat` を叩き、モック Retrieval/LLM で 200 と JSON スキーマ整合をアサート。

### 16.3 非機能テスト

* **ロギング**: `logging.level` 切り替え時に DEBUG ログが出ることをキャプチャして確認。

### 16.4 テストデータ運用

* `tests/fixtures/egov_small/` に最小限の XML/JSONL/チャンク済みデータをコミットし、全テストはこれのみを使用。外部ダウンロードはスキップ対象として明示。
* lawqa_jp も 3〜5 問のサブセットを `tests/fixtures/lawqa_small.json` として同梱し、eval テストで利用。

### 16.5 実行コマンド例

```bash
make test                       # uv run python -m pytest
uv run python -m pytest -k chunk  # 部分実行（直接叩く場合）
```

* CI は `-m "not external"` をデフォルトにし、外部API/ダウンロード系だけをマーカーで除外する。

---

## 17. まとめ

ポイント：

* **構造付き中間表現（LawNode）を必ず作る**
  → チャンキング戦略を切り替えやすくするため
* Qdrant は **dense + sparse named vectors** でハイブリッド検索できる前提([qdrant.tech][1])
* LLM 呼び出しは **OpenAI Responses API 準拠**（本家 or LM Studio gpt-oss）([OpenAI Platform][3])
* すべての構成要素（チャンク・埋め込み・retriever・HyDE・rerank）は **config.yaml で切り替え** できる構造にする
* eval と serve は同じ RAG パイプラインを使い回す

## 17. 参考情報

[1]: https://qdrant.tech/course/essentials/day-3/sparse-vectors/?utm_source=chatgpt.com "Sparse Vectors and Inverted Indexes - Qdrant"
[2]: https://huggingface.co/BAAI/bge-reranker-v2-m3?utm_source=chatgpt.com "BAAI/bge-reranker-v2-m3 · Hugging Face"
[3]: https://platform.openai.com/docs/guides/latest-model?utm_source=chatgpt.com "Using GPT-5.1 - OpenAI API"
[4]: https://platform.openai.com/docs/models/text-embedding-3-large?utm_source=chatgpt.com "Model - OpenAI API"
[5]: https://huggingface.co/naver/splade-v3?utm_source=chatgpt.com "naver/splade-v3 · Hugging Face"
[6]: https://note.motnote.com/2025/09/07/gpt-oss/?utm_source=chatgpt.com "OpenAI の「gpt-oss-20b」を LM Studio で使ってみた | MOT ..."
[7]: https://platform.openai.com/docs/guides/embeddings?utm_source=chatgpt.com "Vector embeddings - OpenAI API"
[8]: https://qdrant.tech/documentation/advanced-tutorials/using-multivector-representations/?utm_source=chatgpt.com "How to Use Multivector Representations with Qdrant ..."
[9]: https://qdrant.tech/course/essentials/day-1/embedding-models/?utm_source=chatgpt.com "Points, Vectors and Payloads - Qdrant"
[10]: https://qdrant.tech/articles/vector-search-resource-optimization/?utm_source=chatgpt.com "Vector Search Resource Optimization Guide - Qdrant"
[11]: https://qdrant.tech/documentation/concepts/vectors/?utm_source=chatgpt.com "Vectors - Qdrant"
