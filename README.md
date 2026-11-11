# RAG Practice

使用 Gemini + Chroma 實作的 RAG（檢索增強生成）練習專案。支援純文字與 PDF、Word、PPT、Excel、HTML、Markdown 等文件，回答會附上資料出處。

## 功能

- **文件解析**：`.txt` 依標點切分；其他格式透過 [Docling](https://github.com/docling-project/docling) 依文件結構（章節、表格）切分
- **切塊**：依句子組合成 ≤ 500 字的片段，相鄰片段以整句重疊約 15%
- **增量索引**：掃描 `docs/` 資料夾，只重建有變動的檔案，並移除已刪除檔案的索引
- **出處標註**：回答標註 `[1]`，並列出來源檔名、章節與頁碼
- **對話記憶**：追問時會依最近 3 輪對話改寫問題後再檢索
- **錯誤處理**：API 暫時性錯誤自動重試；每日配額用完時立即停止

## 專案結構

```
├── Embedding.py              # 核心：切塊、embedding、索引、檢索、回答
├── RAG.py                    # 互動式問答
├── eval_retrieval.py         # 檢索品質評估（Recall@k、MRR）
├── docs/                     # 知識庫文件放這裡
├── eval/eval_set.json        # 評估題目
├── tests/                    # 單元測試
├── examples/
│   └── gemini_file_search.py # 參考：Gemini 官方 File Search 託管做法
├── flowise/
│   └── rag_agents.json       # 參考：Flowise AgentFlow（自我修正 RAG 流程）
├── .env.example              # 環境變數範本
└── requirements.txt
```

## 安裝

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 填入 GEMINI_API_KEY
```

## 使用

```bash
python RAG.py                 # 互動問答（輸入 q 離開、r 清除對話紀錄）
python eval_retrieval.py      # 評估檢索品質（只呼叫 embedding API）
python -m pytest              # 執行測試（不需要 API Key）
```

把要查詢的文件放進 `docs/`，執行時會自動建立索引。

## 設定（`.env`）

| 變數 | 預設值 | 說明 |
|---|---|---|
| `GEMINI_API_KEY` | （必填） | Gemini API 金鑰 |
| `EMBED_MODEL` | `models/gemini-embedding-2` | Embedding 模型 |
| `EMBED_DIM` | `768` | 向量維度 |
| `LLM_MODEL` | `gemini-3.6-flash` | 回答用的模型 |
| `CHROMA_PATH` | `./chroma_db` | 向量資料庫位置 |
| `COLLECTION_NAME` | `my_rag_collection` | 資料庫名稱前綴（實際名稱會加上模型與維度） |
| `DOCS_DIR` | `docs` | 知識庫資料夾 |

## 注意事項

- 第一次處理 PDF 時，Docling 會下載版面分析與 OCR 模型（約數百 MB）
- 更換 `EMBED_MODEL` 或 `EMBED_DIM` 會使用新的資料庫，所有文件會重新建立索引
