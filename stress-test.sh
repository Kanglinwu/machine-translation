#!/bin/bash

# 顯示使用方法
usage() {
    echo "使用方法: $0 [-c 並發數] [-n 請求總數]"
    echo "例如: $0 -c 50 -n 1000"
    echo "選項:"
    echo "  -c    並發數 (預設: 50)"
    echo "  -n    請求總數 (預設: 1000)"
    echo "  -h    顯示此幫助信息"
    exit 1
}

# 設置預設值
CONCURRENT=50
REQUESTS=1000

# 解析命令行參數
while getopts "c:n:h" opt; do
    case $opt in
        c) CONCURRENT=$OPTARG ;;
        n) REQUESTS=$OPTARG ;;
        h) usage ;;
        ?) usage ;;
    esac
done

# 驗證參數
if ! [[ "$CONCURRENT" =~ ^[0-9]+$ ]] || ! [[ "$REQUESTS" =~ ^[0-9]+$ ]]; then
    echo "錯誤: 並發數和請求總數必須是正整數"
    usage
fi

if [ "$CONCURRENT" -gt "$REQUESTS" ]; then
    echo "錯誤: 並發數不能大於請求總數"
    usage
fi

# 配置參數
HOST="10.10.10.95:2486"
ENDPOINT="/translate"
JSON_FILE="payload.json"
RESULT_FILE="test_result.txt"

# 創建測試用的 JSON 文件
cat > "$JSON_FILE" << EOF
{"msg": "Cô gái này đẹp quá!", "target_lang": "en"}
EOF

# 顯示測試開始信息
echo "開始進行壓力測試..."
echo "目標 URL: http://$HOST$ENDPOINT"
echo "並發數: $CONCURRENT"
echo "總請求數: $REQUESTS"
echo "----------------------------"

# 執行測試並保存結果
ab -n $REQUESTS -c $CONCURRENT \
   -T 'application/json' \
   -p "$JSON_FILE" \
   -H "Content-Type: application/json" \
   "http://$HOST$ENDPOINT" > "$RESULT_FILE"

# 解析並顯示測試結果
echo ""
echo "測試結果摘要："
echo "----------------------------"

# 提取關鍵指標
COMPLETED=$(grep "Complete requests:" "$RESULT_FILE" | awk '{print $3}')
FAILED=$(grep "Failed requests:" "$RESULT_FILE" | awk '{print $3}')
RPS=$(grep "Requests per second:" "$RESULT_FILE" | awk '{print $4}')
MEAN_TIME=$(grep "Time per request:" "$RESULT_FILE" | head -1 | awk '{print $4}')
P95_TIME=$(grep "95%" "$RESULT_FILE" | awk '{print $2}')
SUCCESS_RATE=$(awk "BEGIN {print (($COMPLETED-$FAILED)/$COMPLETED)*100}")

# 顯示結果
echo "完成請求數: $COMPLETED"
echo "失敗請求數: $FAILED"
echo "成功率: ${SUCCESS_RATE}%"
echo "每秒請求數 (RPS): $RPS"
echo "平均響應時間: ${MEAN_TIME}ms"
echo "95%請求響應時間: ${P95_TIME}ms"

# 清理臨時文件
rm -f "$JSON_FILE"

# 顯示完整報告位置
echo ""
echo "完整測試報告已保存至: $RESULT_FILE"
