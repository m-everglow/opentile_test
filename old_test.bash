#!/bin/bash
# 用法: ./run_test.bash [pr=<PR编号>] [test_num=<算子编号>] [no_compile]
# 功能：编译并运行测试（并发数固定为 20）。
#   - 若未指定 pr，则使用 main 分支；
#   - 若不指定 test_num，则运行 oldtest + 所有算子；
#   - 若指定 test_num，则仅运行对应编号的算子（如 test_num=2 -> 跑 02_xxx 或 2_xxx）；
#   - 若指定 no_compile，则跳过编译，直接使用系统 PATH 中的 opentileas 运行；
#   - 当 main 分支已是最新且 build/opentileas 可用时，自动跳过编译；
#   - 若 main 最新但 build 为空，自动编译；若 main 落后则更新并编译；
#   - PR 模式结束后自动清除 build 目录，防止污染后续 main 测试。

SECONDS=0
export ASCEND_RT_VISIBLE_DEVICES=5
echo "========== 脚本开始 =========="

# ---- 获取脚本所在目录（绝对路径） ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "脚本目录: $SCRIPT_DIR"

source /data/set_proxy.bash
source /data/setenv.bash

# ---- 仓库锁相关变量 ----
LOCK_FD=""          # 锁文件描述符
LOCK_FILE=""        # 锁文件路径

# ---- 获取基于仓库路径的锁 ----
acquire_repo_lock() {
    local repo_dir="$1"
    # 生成锁文件路径：基于仓库绝对路径的 MD5 哈希
    local lockfile="/tmp/opentileas_build_$(echo "$repo_dir" | md5sum | cut -d' ' -f1).lock"

    # 如果已经持有锁，先释放（防御性）
    if [ -n "$LOCK_FD" ]; then
        release_repo_lock
    fi

    LOCK_FILE="$lockfile"
    # 动态分配文件描述符
    exec {LOCK_FD}>"$LOCK_FILE"
    if ! flock -n "$LOCK_FD"; then
        echo "错误：另一个脚本实例正在使用同一仓库目录 ($repo_dir)，请等待其完成后再试。"
        # 关闭文件描述符
        eval "exec $LOCK_FD>&-"
        LOCK_FD=""
        LOCK_FILE=""
        return 1
    fi
    echo "已获取仓库锁: $LOCK_FILE"
    return 0
}

release_repo_lock() {
    if [ -n "$LOCK_FD" ]; then
        # 释放锁并关闭文件描述符
        flock -u "$LOCK_FD" 2>/dev/null
        eval "exec $LOCK_FD>&-"
        LOCK_FD=""
        LOCK_FILE=""
    fi
}

# ---- 清理函数（仅在确实修改了仓库时才重置） ----
cleanup() {
    # 先释放仓库锁（如果有）
    release_repo_lock

    if ! $PERFORMED_GIT_OPERATIONS; then
        echo "未对仓库做破坏性修改，跳过清理。"
        return
    fi

    echo "正在进行清理..."
    cd "$SCRIPT_DIR/OpenTileAS" 2>/dev/null || return
    # 清除可能残留的 rebase 状态
    git rebase --abort 2>/dev/null
    git checkout main 2>/dev/null || true
    git reset --hard origin/main 2>/dev/null || true

    # reset --hard 不会清理未跟踪文件
    git clean -ffd 2>/dev/null || true

    if [ -n "$BRANCH_NAME" ]; then
        git branch -D "${BRANCH_NAME}" 2>/dev/null || true
    fi

    # PR 模式结束后删除 build 目录，避免污染后续 main 测试
    if $IS_PR_MODE; then
        echo "清除 PR 编译产物..."
        rm -rf "$SCRIPT_DIR/OpenTileAS/build"
    fi

    cd "$SCRIPT_DIR" 2>/dev/null || return
    echo "清理完成"
}
trap cleanup EXIT
# ★ 捕获中断信号，强制终止所有后台子进程
trap 'echo "收到中断信号，正在终止所有子进程..."; kill $(jobs -p) 2>/dev/null; exit 130' INT TERM

# ====================================================
# 参数解析：支持 pr=xxx, test_num=xx, no_compile
# ====================================================
MODE="main"
PR_NUM=""
BRANCH_NAME=""
TEST_NUM=""
NO_COMPILE=false
MAX_PARALLEL=20            # 固定并发数
IS_PR_MODE=false          # 标记是否为 PR 模式
PERFORMED_GIT_OPERATIONS=false

for arg in "$@"; do
    if [[ "$arg" =~ ^pr=([0-9]+)$ ]]; then
        PR_NUM="${BASH_REMATCH[1]}"
        MODE="pr"
    elif [[ "$arg" =~ ^test_num=([0-9]+)$ ]]; then
        TEST_NUM="${BASH_REMATCH[1]}"
    elif [[ "$arg" == "no_compile" ]]; then
        NO_COMPILE=true
    fi
done

# ---- 冲突检查：pr 与 no_compile 不能同时存在 ----
if [ -n "$PR_NUM" ] && $NO_COMPILE; then
    echo "错误：pr 与 no_compile 参数冲突，不能同时指定。"
    exit 1
fi

if [ -n "$PR_NUM" ]; then
    BRANCH_NAME="pr_${PR_NUM}"
    IS_PR_MODE=true
fi

# ====================================================
# 决定是否编译以及如何准备环境
# ====================================================
if $NO_COMPILE; then
    echo "========== no_compile 模式：检查 opentileas 是否可用 =========="
    if ! which opentileas >/dev/null 2>&1; then
        echo "错误：no_compile 模式下未找到 opentileas，请确认已安装并在 PATH 中。"
        exit 1
    fi
    echo "opentileas 已就绪: $(which opentileas)"
else
    # ---------- 需要编译的情况：先获取仓库锁 ----------
    cd "$SCRIPT_DIR/OpenTileAS" || { echo "错误：无法进入 $SCRIPT_DIR/OpenTileAS 目录"; exit 1; }
    # 获取基于该仓库路径的锁，确保同一目录下不会并发运行
    if ! acquire_repo_lock "$SCRIPT_DIR/OpenTileAS"; then
        exit 1
    fi
    # 锁已获取，后续操作（包括 git 更新、编译）均在保护中

    if [ "$MODE" = "pr" ]; then
        echo "正在切换到 main 分支并强制拉取最新代码..."
        git checkout main || exit 1
        git fetch origin main || exit 1
        git reset --hard origin/main || exit 1
        git clean -ffd || exit 1
        PERFORMED_GIT_OPERATIONS=true

        echo "正在删除可能存在的本地分支 ${BRANCH_NAME} ..."
        git branch -D "${BRANCH_NAME}" 2>/dev/null || true

        echo "正在从远程拉取 PR #${PR_NUM} 的代码..."
        git fetch https://gitcode.com/OpenTileIR/OpenTileAS.git "+refs/merge-requests/${PR_NUM}/head:${BRANCH_NAME}" || exit 1

        echo "正在切换到 ${BRANCH_NAME} 分支..."
        git checkout "${BRANCH_NAME}" || exit 1

        echo "正在将 PR 分支 rebase 到最新的 origin/main 上..."
        git rebase --abort 2>/dev/null || true
        git rebase origin/main || {
            echo "错误：rebase 失败，可能存在冲突，请手动解决"
            exit 1
        }
        PERFORMED_GIT_OPERATIONS=true

        echo $(git rev-parse HEAD)

        echo "正在清理旧的 build 目录..."
        rm -rf build
        mkdir build
        cd build || exit 1

        echo "正在执行 cmake ..."
        cmake .. || exit 1

        echo "正在执行 make -j48 ..."
        make -j48 || exit 1

        export PATH=/data/y00939135/test/OpenTileAS/build/bin:$PATH

    elif [ "$MODE" = "main" ]; then
        echo "正在检查 main 分支是否需要更新..."
        git checkout main || exit 1
        git fetch origin main || exit 1

        LOCAL_HASH=$(git rev-parse HEAD)
        REMOTE_HASH=$(git rev-parse origin/main)

        if [ "$LOCAL_HASH" = "$REMOTE_HASH" ]; then
            echo "本地 main 分支已是最新。"
            cd "$SCRIPT_DIR/OpenTileAS" || exit 1
            if [ -x build/bin/opentileas ]; then
                echo "已有可用编译产物，跳过编译。"
                export PATH="$SCRIPT_DIR/OpenTileAS/build/bin:$PATH"
            else
                echo "未找到编译产物，将进行编译（本地代码已最新）..."
                rm -rf build
                mkdir build
                cd build || exit 1
                echo "正在执行 cmake ..."
                cmake .. || exit 1
                echo "正在执行 make -j48 ..."
                make -j48 || exit 1
                export PATH="$SCRIPT_DIR/OpenTileAS/build/bin:$PATH"
            fi
        else
            echo "本地 main 分支落后于远端，需要更新并重新编译..."
            git reset --hard origin/main || exit 1
            git clean -ffd || exit 1
            PERFORMED_GIT_OPERATIONS=true

            echo "正在清理旧的 build 目录..."
            rm -rf build
            mkdir build
            cd build || exit 1

            echo "正在执行 cmake ..."
            cmake .. || exit 1

            echo "正在执行 make -j48 ..."
            make -j48 || exit 1

            export PATH=/data/y00939135/test/OpenTileAS/build/bin:$PATH
        fi
    fi
fi

# ---- 返回脚本目录 ----
cd "$SCRIPT_DIR" || exit 1

echo "========== 环境准备完成 =========="

# ====================================================
# 定义测试运行函数（原逻辑，无并发控制）
# ====================================================
run_test() {
    local file="$1"
    local max_retries=5
    local retry_count=0
    local tmp_out=$(mktemp)
    local exit_code=0

    while [ $retry_count -lt $max_retries ]; do
        > "$tmp_out"
        if [[ "$file" == *"fa.py" ]]; then
            TRITON_ALWAYS_COMPILE=1 python "$file" > "$tmp_out" 2>&1
        else
            TRITON_ALWAYS_COMPILE=1 python -m pytest -v --tb=short "$file" > "$tmp_out" 2>&1
        fi
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
	    rm -f "$tmp_out"
	    return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "⚠️ 第 $retry_count 次运行失败，重试中... ($file)" >&2
            fi
        fi
    done

    # 所有重试都失败，将日志输出到 stderr 以便上层捕获
    echo "❌ $file 运行失败 (退出码 $exit_code)，已重试 $max_retries 次" >&2
    cat "$tmp_out" >&2
    rm -f "$tmp_out"
    return 1
}

# test vffusion max_parallel
run_test_max_parallel() {
    local file="$1"
    local max_retries=5
    local retry_count=0
    local tmp_out=$(mktemp)
    local exit_code=0

    echo "############test ($file) max_parallel#############"
    while [ $retry_count -lt $max_retries ]; do
        > "$tmp_out"
        TRITON_ALWAYS_COMPILE=1 FUSION_MODE=max_parallel python -m pytest -v --tb=short "$file" > "$tmp_out" 2>&1
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            rm -f "$tmp_out"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "⚠️ 第 $retry_count 次运行失败，重试中... ($file)" >&2
            fi
        fi
    done

    # 所有重试都失败，将日志输出到 stderr 以便上层捕获
    echo "❌ $file 运行失败 (退出码 $exit_code)，已重试 $max_retries 次" >&2
    cat "$tmp_out" >&2
    rm -f "$tmp_out"
    return 1
}
# ====================================================
# 并发执行测试框架
# ====================================================
# 使用临时目录存放结果文件
RESULT_DIR=$(mktemp -d -t test_results_XXXXXX)
echo "并发测试结果目录: $RESULT_DIR"
echo "最大并发数: $MAX_PARALLEL (固定)"

# 启动一个测试任务，参数：任务唯一ID、测试文件路径
launch_test() {
    local id="$1"
    local test_file="$2"
    local log_file="${RESULT_DIR}/${id}.log"
    local status_file="${RESULT_DIR}/${id}.status"
    local time_file="${RESULT_DIR}/${id}.time"

    (
        # ★ 关闭从父进程继承的文件锁，防止锁泄露
        if [ -n "$LOCK_FD" ]; then
            eval "exec $LOCK_FD>&-" 2>/dev/null
            LOCK_FD=""
        fi
        START_TIME=$(date +%s)
        if run_test "$test_file" >> "$log_file" 2>&1; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "$ELAPSED" > "$time_file"
            echo "0" > "$status_file"
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "$ELAPSED" > "$time_file"
            echo "1" > "$status_file"
        fi
        if [[ "$test_file" == *"46_test_rmsnorm/ci.py" ]]; then
            if run_test_max_parallel "$test_file" >> "$log_file" 2>&1; then
                END_TIME=$(date +%s)
                ELAPSED=$((END_TIME - START_TIME))
                echo "$ELAPSED" > "$time_file"
                echo "0" > "$status_file"
            else
                END_TIME=$(date +%s)
                ELAPSED=$((END_TIME - START_TIME))
                echo "$ELAPSED" > "$time_file"
                echo "1" > "$status_file"
            fi
	fi
    ) &
}

# 收集指定状态文件对应的测试结果（0 通过，1 失败，其他为跳过）
collect_result() {
    local id="$1"
    local status_file="${RESULT_DIR}/${id}.status"
    if [ -f "$status_file" ]; then
        local st=$(cat "$status_file")
        if [ "$st" = "0" ]; then
            echo "PASS"
        else
            echo "FAIL"
        fi
    else
        echo "SKIP"
    fi
}

# ====================================================
# 准备待测任务列表（统一管理 oldtest + 算子）
# ====================================================
declare -A TASK_MAP      # key: 唯一ID, value: 测试文件路径
declare -A TASK_TYPE     # key: 唯一ID, value: "oldtest" 或 "operator:dirname"
task_count=0
skipped_dirs=()

# 1) oldtest 部分（仅在未指定 test_num 时添加）
if [ -z "$TEST_NUM" ]; then
    old_files=(
        "testcase/oldtest/test_gmm_fwd.py"
        "testcase/oldtest/test_fused_ce_forward_backward_diff.py"
        "testcase/oldtest/fa_all.py"
    )
    for file in "${old_files[@]}"; do
        id="oldtest_${task_count}"
        TASK_MAP[$id]="$file"
        TASK_TYPE[$id]="oldtest"
        ((task_count++))
    done
fi

# 2) 算子部分：扫描 testcase 下的目录
cd "$SCRIPT_DIR/testcase" || { echo "错误：无法进入 testcase 目录"; exit 1; }
shopt -s nullglob
test_dirs=( [0-9]*_*/ )
shopt -u nullglob

if [ ${#test_dirs[@]} -eq 0 ]; then
    echo "错误：testcase 下没有找到算子目录（格式如 01_xxx/）"
    exit 1
fi

IFS=$'\n' test_dirs=($(sort <<<"${test_dirs[*]}"))
unset IFS

cd "$SCRIPT_DIR" || exit 1

for dir in "${test_dirs[@]}"; do
    dir="${dir%/}"
    dir_num="${dir%%_*}"
    # 如果指定了 test_num，只处理编号匹配的目录
    if [ -n "$TEST_NUM" ] && [ "$dir_num" -ne "$TEST_NUM" ] 2>/dev/null; then
        continue
    fi

    test_file="testcase/$dir/ci.py"
    if [ -f "$test_file" ]; then
        id="op_${dir_num}_${dir}"
        TASK_MAP[$id]="$test_file"
        TASK_TYPE[$id]="operator:${dir}"
        ((task_count++))
    else
        # 没有 ci.py 的目录直接记入跳过列表（不参与并发）
        skipped_dirs+=("$dir")
    fi
done

echo "共收集到 $task_count 个测试任务，跳过 ${#skipped_dirs[@]} 个（无 ci.py）"

# ====================================================
# 并发执行所有任务
# ====================================================
active_jobs=0
task_ids=("${!TASK_MAP[@]}")

for id in "${task_ids[@]}"; do
    test_file="${TASK_MAP[$id]}"

    # 控制并发数：当活跃作业数达到上限时，等待任意一个完成
    while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 0.5
    done

    echo "启动测试: $id ($test_file)"
    launch_test "$id" "$test_file"
done

# 等待所有后台作业完成
wait

# ====================================================
# 在所有任务完成后，先打印错误详情，再进行统计
# ====================================================
echo ""
echo "========== 失败任务详细日志 =========="
any_fail=false
for id in "${task_ids[@]}"; do
    result=$(collect_result "$id")
    if [ "$result" = "FAIL" ]; then
        any_fail=true
        log_file="${RESULT_DIR}/${id}.log"
        echo "--------- 失败任务: $id ---------"
        echo "文件: ${TASK_MAP[$id]}"
        echo "--------- 日志（最后 50 行）---------"
        tail -n 50 "$log_file"
        echo ""
        echo "完整日志文件: $log_file"
        echo "========================================"
    fi
done

if ! $any_fail; then
    echo "没有失败的任务。"
fi
echo ""

# ====================================================
# 输出每个用例的耗时，并标记超时（> 5 分钟）
# ====================================================
echo "========== 每个用例耗时 =========="
timeout_count=0
timeout_list=()
for id in "${task_ids[@]}"; do
    time_file="${RESULT_DIR}/${id}.time"
    if [ -f "$time_file" ]; then
        elapsed=$(cat "$time_file")
        minutes=$((elapsed / 60))
        seconds=$((elapsed % 60))
        echo -n "$id : ${minutes}m ${seconds}s"
        if [ $elapsed -gt 300 ]; then
            echo "  ⚠️ 超时（超过5分钟）"
            timeout_list+=("$id")
            ((timeout_count++))
        else
            echo ""
        fi
    else
        echo "$id : 未记录耗时"
    fi
done
echo ""

# ====================================================
# 汇总结果统计
# ====================================================
passed_dirs=()
failed_dirs=()
failed_oldtest=()
failed_count=0

for id in "${task_ids[@]}"; do
    result=$(collect_result "$id")
    test_file="${TASK_MAP[$id]}"
    type_info="${TASK_TYPE[$id]}"

    case "$result" in
        PASS)
            if [[ "$type_info" == operator:* ]]; then
                dir_name="${type_info#operator:}"
                passed_dirs+=("$dir_name")
            fi
            ;;
        FAIL)
            ((failed_count++))
            if [[ "$type_info" == oldtest ]]; then
                failed_oldtest+=("$test_file")
            else
                dir_name="${type_info#operator:}"
                failed_dirs+=("$dir_name")
            fi
            ;;
        SKIP)
            ;;
    esac
done

# ====================================================
# 输出统计信息
# ====================================================
echo "========================================"
if [ -n "$TEST_NUM" ]; then
    echo "仅运行算子编号 ${TEST_NUM} 的测试结果："
else
    echo "所有测试统计："
fi
echo "  通过: ${#passed_dirs[@]} (算子) + oldtest 通过数未单独统计"
echo "  失败: $failed_count"
echo "  跳过: ${#skipped_dirs[@]} (未找到 ci.py)"
if [ ${#failed_dirs[@]} -gt 0 ]; then
    echo "失败算子列表:"
    for d in "${failed_dirs[@]}"; do
        echo "  - $d"
    done
fi
if [ ${#failed_oldtest[@]} -gt 0 ]; then
    echo "失败 oldtest 列表:"
    for f in "${failed_oldtest[@]}"; do
        echo "  - $f"
    done
fi
if [ ${#skipped_dirs[@]} -gt 0 ]; then
    echo "跳过算子列表:"
    for d in "${skipped_dirs[@]}"; do
        echo "  - $d"
    done
fi
if [ $timeout_count -gt 0 ]; then
    echo "超时用例数: $timeout_count"
    echo "超时用例列表:"
    for t in "${timeout_list[@]}"; do
        echo "  - $t"
    done
fi
echo "========================================"

# 清理结果目录
rm -rf "$RESULT_DIR"

echo "------------------------"
if [ $failed_count -eq 0 ]; then
    echo "所有测试均通过 ✅"
else
    echo "失败的测试总数: $failed_count"
fi

echo "========Total time: $((SECONDS/60))min $((SECONDS%60))sec ========"

if [ $failed_count -eq 0 ]; then
    exit 0
else
    exit 1
fi
