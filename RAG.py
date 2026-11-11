from Embedding import DOCS_DIR, HISTORY_TURNS, Embedding

rag = Embedding()

print("檢查索引中...")
rag.index_folder(DOCS_DIR)

history = []
while True:
    q = input("\n請輸入問題（輸入 q 離開，輸入 r 清除對話紀錄）：").strip()
    if q.lower() == "q":
        break
    if q.lower() == "r":
        history.clear()
        print("已清除對話紀錄。")
        continue
    if not q:
        continue
    try:
        a = rag.answer(q, history=history)
    except RuntimeError as e:
        print(e)
        continue
    print(a)
    history.append((q, Embedding.strip_sources(a)))
    del history[:-HISTORY_TURNS]
