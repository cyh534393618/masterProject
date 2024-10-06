import wenet
import soundfile as sf
from jiwer import wer, process_words
import pandas as pd
import os
import re

# 定義一個函數來清理文本
def clean_text(text):
    # 移除首尾空格
    text = text.strip()
    # 使用正則表達式去除所有非字母、非數字的字符（僅保留字母和空格）
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Load validated.tsv file
def load_validated_data(tsv_file):
    df = pd.read_csv(tsv_file, delimiter='\t')

    # filter conditions
    filtered_df = df[
        df['gender'].notna() & (df['gender'] != '') &                  # 性别不为空
        (df['gender'] != 'other') &                                     # 排除 'other'
        (df['up_votes'] > 3) &                                          # up_votes 大于 3
        (df['down_votes'] == 0) &                                       # down_votes 等于 0
        df['age'].notna() & (df['age'] != '') &                        # age 不为空
        df['accent'].notna() & (df['accent'] != '')                    # accent 不为空
    ]

    print(filtered_df.shape)

    return filtered_df

# 加載 WeNet 模型
model = wenet.load_model('english')

max_files = 1000  # 只處理前 1000 條音頻
processed_files = 0  # 独立的计数器，用来控制处理的文件数

# 輸入文件的路徑 (假設你已經下載了 Common Voice 並解壓)
tsv_file = './cv-corpus-18.0-2022-06-14/en/validated.tsv'

df = load_validated_data(tsv_file)

# 初始化 WER 評估結果
total_wer = 0
#total_files = len(df)

# 逐個處理 GigaSpeech 測試文件
for index, row in df.iterrows():
    if processed_files >= max_files:
       break  # 超過 1000 條後停止
    
    audio_file = os.path.join('cv-corpus-18.0-2022-06-14', 'en\clips', row['path'])

    # Check file format, if it is MP3 convert to WAV
    if audio_file.endswith('.mp3'):
       audio_file = audio_file.replace('.mp3', '.wav')

    # 使用 WeNet 模型進行語音轉文字
    result = model.transcribe(audio_file)
    reference_text = clean_text(row['sentence'].upper())
    predicted_text = clean_text(result['text'].replace("▁", " ").upper())

    # 計算當前音頻的 WER
    current_wer = wer(reference_text, predicted_text)
    print(f"Audio file: {audio_file}")
    print(f"Reference: {reference_text}")
    print(f"Predicted: {predicted_text}")
    print(f"WER: {current_wer}\n")

    # 累加 WER
    total_wer += current_wer

    processed_files += 1  # 每处理一个有效音频文件，计数器加 1

# 計算平均 WER
average_wer = total_wer / max_files
print(f"total files: {max_files}")
print(f"Average WER on test set: {average_wer}")

#result = model.transcribe('output_file.wav')
#print(result['text'])