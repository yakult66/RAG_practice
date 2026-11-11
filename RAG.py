from Embedding import Embedding

rag = Embedding()

print("檢查索引中...")
if rag.index_file("sample.txt"):
    print("索引已更新！")
else:
    print("檔案沒有變更，沿用既有索引。")

while True:
    q = input("\n請輸入問題（輸入 q 離開）：")
    if q.strip().lower() == "q":
        break
    if not q.strip():
        continue
    print(rag.answer(q, verbose=False))
