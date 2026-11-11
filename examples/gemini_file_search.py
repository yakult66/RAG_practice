from google import genai
from google.genai import types
import time
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client()

# File name will be visible in citations
file_search_store = client.file_search_stores.create(
    config={
        'display_name': 'your-fileSearchStore-name',
        'embedding_model': 'models/gemini-embedding-2'
    }
)

operation = client.file_search_stores.upload_to_file_search_store(
file='docs/sample.txt',
file_search_store_name=file_search_store.name,
config={
    'display_name' : 'display-file-name',
}
)

print("開始將檔案上傳到雲端建立索引...")

while not operation.done:
    print("系統處理中，請稍等 5 秒鐘...")
    time.sleep(5)
    operation = client.operations.get(operation)

print("檔案處理完成！現在開始向 AI 提問...")
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="""請問我今天吃了什麼?""",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_search_store.name]
                )
            )
        ]
    )
)

print(response.text)
