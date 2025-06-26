# app.py: Flask 應用程式主文件，負責設定伺服器、定義 API 端點，並處理來自前端的請求。它將請求轉發給 video_processing 模組進行實際的影片分析，然後將結果返回給前端。

import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from video_processing import process_video_data

# 初始化 Flask 應用程式：啟用 CORS 並配置日誌記錄。
app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.INFO)

# API 端點：/process_video (POST 請求)，用於處理影片分析請求。
@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    try:
        # 從前端獲取數據：檢查影片檔案、解析捕獲點、鏡像設定和範圍寬度等表單數據。
        if 'video' not in request.files: 
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        captures = json.loads(request.form.get('captures', '[]'))
        mirror_settings = json.loads(request.form.get('mirrorSettings', '{}'))
        range_width = int(request.form.get('rangeWidth', 20))

        if not captures:
             return jsonify({"error": "No capture timestamps provided"}), 400

        app.logger.info(f"Received {len(captures)} captures, range width {range_width}, and mirror settings: {mirror_settings}")

        # 調用 video_processing 模組中的函數來處理影片數據。
        processed_angles = process_video_data(video_file, captures, mirror_settings, range_width)

        # 記錄成功處理的信息並返回結果。
        app.logger.info("Successfully processed all captures. Returning data.")
        return jsonify(processed_angles)

    except Exception as e:
        # 捕獲並記錄任何未預期的錯誤，然後返回內部伺服器錯誤響應。
        app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# 運行伺服器：當此腳本直接運行時，啟動 Flask 開發伺服器 (debug=True, port=5002)。
if __name__ == '__main__':
    app.run(debug=True, port=5002)
