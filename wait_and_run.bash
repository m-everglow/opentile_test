#!/bin/bash
# 用法: ./wait_and_run.bash pr=370 [test_num=2]
# 功能：轮询检测 run_test.bash 使用的全局锁 (/tmp/OpenTileAS_build.lock)，
#       一旦空闲，立即自动执行 run_test.bash 并传入你给的参数。
#       如果抢锁时被别人抢先（极少发生的竞争情况），会自动继续等待重试，
#       而不是直接报错退出。

LOCKFILE="/tmp/OpenTileAS_build.lock"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_test.bash"

INTERVAL=5        # 轮询间隔（秒），可按需调整
RACE_THRESHOLD=5   # 若运行耗时小于此秒数就退出，视为"抢锁失败"，不算真正跑完

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "错误：未找到 $RUN_SCRIPT，请确认该脚本与 run_test.bash 放在同一目录下。"
    exit 1
fi

echo "========== 等待脚本启动 =========="
echo "监控锁文件: $LOCKFILE"
echo "轮询间隔: ${INTERVAL}s"
echo "一旦锁空闲，将自动执行: bash run_test.bash $*"
echo "===================================="

waited=0

while true; do
    # 用独立的 fd(201) 检测锁是否空闲，检测完立即释放，
    # 不影响 run_test.bash 自己稍后用 fd 200 加的真实锁。
    exec 201>"$LOCKFILE"
    if flock -n 201; then
        flock -u 201
        exec 201>&-

        echo "[$(date '+%H:%M:%S')] 检测到锁空闲，尝试启动 run_test.bash ..."
        start_ts=$(date +%s)

        bash "$RUN_SCRIPT" "$@"
        code=$?

        elapsed=$(( $(date +%s) - start_ts ))

        if [ $code -eq 0 ]; then
            echo "✅ run_test.bash 执行成功（耗时 ${elapsed}s）"
            exit 0
        fi

        if [ $elapsed -lt $RACE_THRESHOLD ]; then
            # 跑得太快就失败了，大概率是检测到空闲后被别人抢先拿到了锁，
            # 而不是真正的测试失败，因此继续等待重试。
            echo "⚠️ 疑似抢锁失败（运行仅 ${elapsed}s 就退出），继续等待重试..."
        else
            echo "❌ run_test.bash 运行失败（退出码 $code，耗时 ${elapsed}s），不再重试。"
            exit $code
        fi
    else
        exec 201>&-
        waited=$((waited + INTERVAL))
        echo "[$(date '+%H:%M:%S')] 锁仍被占用，已等待 ${waited}s，${INTERVAL}s 后重试..."
    fi

    sleep "$INTERVAL"
done
