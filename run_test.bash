#!/bin/bash
# 用法: ./run_test.bash [pr=<PR编号>] [test_num=<算子编号>] [npu=<0-7>] [no_compile]
# 功能：编译并运行测试（并发数固定为 20）。
#   - 若未指定 pr，则使用 main 分支；
#   - 若不指定 test_num，则运行 oldtest + 所有算子；
#   - 若指定 test_num，则仅运行对应编号的算子（如 test_num=2 -> 跑 02_xxx 或 2_xxx）；
#   - 若指定 npu，则使用指定 NPU（0-7）；未指定时默认使用 NPU 4；
#   - 若指定 no_compile，则跳过编译，直接使用系统 PATH 中的 opentileas 运行；
#   - 当 main 分支已是最新且 build/opentileas 可用时，自动跳过编译；
#   - 若 main 最新但 build 为空，自动编译；若 main 落后则更新并编译；
#   - PR 模式结束后自动清除 build 目录，防止污染后续 main 测试；
#   - 单个算子运行超过 5 分钟时不终止，允许其继续运行，并在最终汇总中列出；
#   - 只有整个脚本运行超过 15 分钟时才终止仍在运行的进程并报超时错误。

# ====================================================
# 全脚本 15 分钟硬超时
# 使用 GNU timeout 包裹脚本自身，确保编译阶段卡住时也能被终止。
# USR1 用于区分“全脚本超时”和普通 INT/TERM 中断。
# ====================================================
if [ "${RUN_TEST_GLOBAL_TIMEOUT_WRAPPED:-0}" != "1" ]; then
    SELF_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
    if ! command -v timeout >/dev/null 2>&1; then
        echo "错误：未找到 GNU timeout，无法启用 15 分钟全脚本超时保护。"
        exit 1
    fi
    exec env RUN_TEST_GLOBAL_TIMEOUT_WRAPPED=1 \
        timeout --signal=USR1 --kill-after=10s 900s \
        bash "$SELF_PATH" "$@"
fi

SECONDS=0
echo "========== 脚本开始 =========="

# ---- 获取脚本所在目录（绝对路径） ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "脚本目录: $SCRIPT_DIR"

# ====================================================
# 全局状态变量
# ====================================================
LOCK_FD=""
LOCK_FILE=""
MODE="main"
PR_NUM=""
BRANCH_NAME=""
TEST_NUM=""
NPU_DEVICE="4"              # 默认使用 NPU 4
NO_COMPILE=false
MAX_PARALLEL=20
IS_PR_MODE=false
PERFORMED_GIT_OPERATIONS=false
RESULT_DIR=""
active_jobs=0

# 后台测试进程跟踪
# key: pid, value: 1 / task id
declare -A ACTIVE_PIDS
declare -A PID_TASK
declare -A TASK_PID
declare -A TASK_START

# ====================================================
# 参数解析：支持 pr=xxx, test_num=xx, npu=0-7, no_compile
# 参数在任何 git / 编译操作之前校验。
# ====================================================
for arg in "$@"; do
    if [[ "$arg" =~ ^pr=([0-9]+)$ ]]; then
        PR_NUM="${BASH_REMATCH[1]}"
        MODE="pr"
    elif [[ "$arg" =~ ^test_num=([0-9]+)$ ]]; then
        TEST_NUM="${BASH_REMATCH[1]}"
    elif [[ "$arg" =~ ^npu=([0-7])$ ]]; then
        NPU_DEVICE="${BASH_REMATCH[1]}"
    elif [[ "$arg" == npu=* ]]; then
        echo "错误：npu 参数必须是 0-7 之间的整数，例如 npu=0 或 npu=7。"
        exit 1
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

# ---- 根据 npu 参数设置运行设备 ----
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICE"
echo "使用 NPU: $ASCEND_RT_VISIBLE_DEVICES"

source /data/set_proxy.bash
source /data/setenv.bash

# ---- 获取基于仓库路径的锁 ----
acquire_repo_lock() {
    local repo_dir="$1"
    local lockfile="/tmp/opentileas_build_$(echo "$repo_dir" | md5sum | cut -d' ' -f1).lock"

    if [ -n "$LOCK_FD" ]; then
        release_repo_lock
    fi

    LOCK_FILE="$lockfile"
    exec {LOCK_FD}>"$LOCK_FILE"
    if ! flock -n "$LOCK_FD"; then
        echo "错误：另一个脚本实例正在使用同一仓库目录 ($repo_dir)，请等待其完成后再试。"
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
        flock -u "$LOCK_FD" 2>/dev/null
        eval "exec $LOCK_FD>&-"
        LOCK_FD=""
        LOCK_FILE=""
    fi
}

# ====================================================
# 终止所有正在运行的测试进程组
# 每个后台测试任务在 set -m 后拥有独立进程组，因此可以同时终止
# pytest / Python / 其普通子进程，避免只杀掉外层 shell。
# ====================================================
terminate_all_test_processes() {
    local pid
    local pids=()

    for pid in "${!ACTIVE_PIDS[@]}"; do
        pids+=("$pid")
        kill -TERM -- "-$pid" 2>/dev/null || true
    done

    if [ ${#pids[@]} -gt 0 ]; then
        sleep 1
        for pid in "${pids[@]}"; do
            kill -KILL -- "-$pid" 2>/dev/null || true
        done
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
}

handle_interrupt() {
    trap - INT TERM
    echo "收到中断信号，正在终止所有测试子进程..."
    terminate_all_test_processes
    exit 130
}

handle_script_timeout() {
    trap - USR1

    echo ""
    echo "========================================"
    echo "❌ 超时错误：脚本总运行时间超过 15 分钟。"

    if [ "$active_jobs" -gt 0 ]; then
        echo "超时时仍在运行的测试任务："
        local now pid id elapsed
        now=$(date +%s)
        for pid in "${!ACTIVE_PIDS[@]}"; do
            id="${PID_TASK[$pid]}"
            if [ -n "${TASK_START[$id]:-}" ]; then
                elapsed=$((now - TASK_START[$id]))
                echo "  - $id (${TASK_MAP[$id]:-unknown})，已运行 ${elapsed}s"
            else
                echo "  - $id (${TASK_MAP[$id]:-unknown})"
            fi
        done
    fi

    echo "正在终止仍在运行的测试进程..."
    terminate_all_test_processes

    if [ -n "$RESULT_DIR" ] && [ -d "$RESULT_DIR" ]; then
        echo "已保留当前测试日志目录: $RESULT_DIR"
    fi

    echo "========================================"
    exit 124
}

# ====================================================
# 清理函数
# 所有仓库恢复/清理操作完成后才释放仓库锁。
# ====================================================
cleanup() {
    if $PERFORMED_GIT_OPERATIONS; then
        echo "正在进行清理..."

        (
            cd "$SCRIPT_DIR/OpenTileAS" 2>/dev/null || exit 0

            git rebase --abort 2>/dev/null || true
            git checkout main 2>/dev/null || true
            git reset --hard origin/main 2>/dev/null || true
            git clean -ffd 2>/dev/null || true

            if [ -n "$BRANCH_NAME" ]; then
                git branch -D "$BRANCH_NAME" 2>/dev/null || true
            fi

            if $IS_PR_MODE; then
                echo "清除 PR 编译产物..."
                rm -rf "$SCRIPT_DIR/OpenTileAS/build"
            fi
        )

        echo "清理完成"
    else
        echo "未对仓库做破坏性修改，跳过清理。"
    fi

    release_repo_lock
}

trap cleanup EXIT
trap handle_interrupt INT TERM
trap handle_script_timeout USR1

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
    cd "$SCRIPT_DIR/OpenTileAS" || {
        echo "错误：无法进入 $SCRIPT_DIR/OpenTileAS 目录"
        exit 1
    }

    if ! acquire_repo_lock "$SCRIPT_DIR/OpenTileAS"; then
        exit 1
    fi

    if [ "$MODE" = "pr" ]; then
        echo "正在切换到 main 分支并强制拉取最新代码..."
        git checkout main || exit 1
        git fetch origin main || exit 1
        git reset --hard origin/main || exit 1
        git clean -ffd || exit 1
        PERFORMED_GIT_OPERATIONS=true

        echo "正在删除可能存在的本地分支 ${BRANCH_NAME} ..."
        git branch -D "$BRANCH_NAME" 2>/dev/null || true

        echo "正在从远程拉取 PR #${PR_NUM} 的代码..."
        git fetch https://gitcode.com/OpenTileIR/OpenTileAS.git \
            "+refs/merge-requests/${PR_NUM}/head:${BRANCH_NAME}" || exit 1

        echo "正在切换到 ${BRANCH_NAME} 分支..."
        git checkout "$BRANCH_NAME" || exit 1

        echo "正在将 PR 分支 rebase 到最新的 origin/main 上..."
        git rebase --abort 2>/dev/null || true
        git rebase origin/main || {
            echo "错误：rebase 失败，可能存在冲突，请手动解决"
            exit 1
        }
        PERFORMED_GIT_OPERATIONS=true

        echo "$(git rev-parse HEAD)"

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
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"

# ====================================================
# 定义测试运行函数
# ====================================================
run_test() {
    local file="$1"
    local max_retries=5
    local retry_count=0
    local tmp_out
    tmp_out=$(mktemp)
    local exit_code=0

    while [ "$retry_count" -lt "$max_retries" ]; do
        > "$tmp_out"

        if [[ "$file" == *"fa.py" ]]; then
            TRITON_ALWAYS_COMPILE=1 python "$file" > "$tmp_out" 2>&1
        else
            TRITON_ALWAYS_COMPILE=1 python -m pytest -v --tb=short "$file" > "$tmp_out" 2>&1
        fi

        exit_code=$?

        if [ "$exit_code" -eq 0 ]; then
            rm -f "$tmp_out"
            return 0
        fi

        retry_count=$((retry_count + 1))
        if [ "$retry_count" -lt "$max_retries" ]; then
            echo "⚠️ 第 $retry_count 次运行失败，重试中... ($file)" >&2
        fi
    done

    echo "❌ $file 运行失败 (退出码 $exit_code)，已重试 $max_retries 次" >&2
    cat "$tmp_out" >&2
    rm -f "$tmp_out"
    return 1
}

max_parallel_files=(
    "testcase/oldtest/test_gmm_fwd.py"
    "testcase/oldtest/fa_all.py"
    "testcase/01__binned_copy/ci.py"
    "testcase/02__binned_copy_wgrad/ci.py"
    "testcase/03__silu_forward_backward_diff/ci.py"
    "testcase/04__c_conv_varlen/ci.py"
    "testcase/05__fused_ce_forward_backward_diff/ci.py"
    "testcase/11_test_reject_sampling/ci.py"
    "testcase/12_test_magic_reject_sampling/ci.py"
    "testcase/13_test_store_lowrank/ci.py"
    "testcase/15__padded_copy/ci.py"
    "testcase/16__padded_copy_wgrad/ci.py"
    "testcase/17_test_gelu/ci.py"
    "testcase/18_test_silu/ci.py"
    "testcase/19_test_swiglu/ci.py"
    "testcase/22_fused_matmul_bwd_w_kernel/ci.py"
    "testcase/23_test_store_paged_kv/ci.py"
    "testcase/25_chunk_gated_delta_rule_bwd_dhu/ci.py"
    "testcase/28__apply_penalties_temp/ci.py"
    "testcase/32_layer_norm_gated_fwd/ci.py"
    "testcase/36_chunk_gated_delta_rule_fwd_h/ci.py"
    "testcase/37_fused_matmul_bwd_b_kernel/ci.py"
    "testcase/38_fused_matmul_bwd_x_kernel/ci.py"
    "testcase/40_chunk_local_cumsum/ci.py"
    "testcase/45__rmsnorm_forward_backward_diff/ci.py"
    "testcase/46_test_rmsnorm/ci.py"
    "testcase/48_test_pos_emb/ci.py"
)

max_parallel_outline_files=(
    "testcase/oldtest/test_gmm_fwd.py"
    "testcase/oldtest/fa_all.py"
    "testcase/01__binned_copy/ci.py"
    "testcase/02__binned_copy_wgrad/ci.py"
    "testcase/03__silu_forward_backward_diff/ci.py"
    "testcase/04__c_conv_varlen/ci.py"
    "testcase/05__fused_ce_forward_backward_diff/ci.py"
    "testcase/11_test_reject_sampling/ci.py"
    "testcase/12_test_magic_reject_sampling/ci.py"
    "testcase/13_test_store_lowrank/ci.py"
    "testcase/15__padded_copy/ci.py"
    "testcase/16__padded_copy_wgrad/ci.py"
    "testcase/17_test_gelu/ci.py"
    "testcase/18_test_silu/ci.py"
    "testcase/19_test_swiglu/ci.py"
    "testcase/22_fused_matmul_bwd_w_kernel/ci.py"
    "testcase/23_test_store_paged_kv/ci.py"
    "testcase/25_chunk_gated_delta_rule_bwd_dhu/ci.py"
    "testcase/28__apply_penalties_temp/ci.py"
    "testcase/32_layer_norm_gated_fwd/ci.py"
    "testcase/36_chunk_gated_delta_rule_fwd_h/ci.py"
    "testcase/38_fused_matmul_bwd_x_kernel/ci.py"
    "testcase/45__rmsnorm_forward_backward_diff/ci.py"
    "testcase/46_test_rmsnorm/ci.py"
    "testcase/48_test_pos_emb/ci.py"
)

# test vffusion max_parallel
run_test_max_parallel() {
    local file="$1"
    local max_retries=5
    local retry_count=0
    local tmp_out
    tmp_out=$(mktemp)
    local exit_code=0

    echo "############test ($file) max_parallel#############"

    while [ "$retry_count" -lt "$max_retries" ]; do
        > "$tmp_out"

        TRITON_ALWAYS_COMPILE=1 FUSION_MODE=max_parallel \
            python -m pytest -v --tb=short "$file" > "$tmp_out" 2>&1

        exit_code=$?

        if [ "$exit_code" -eq 0 ]; then
            rm -f "$tmp_out"
            return 0
        fi

        retry_count=$((retry_count + 1))
        if [ "$retry_count" -lt "$max_retries" ]; then
            echo "⚠️ 第 $retry_count 次运行失败，重试中... ($file)" >&2
        fi
    done

    echo "❌ $file 运行失败 (退出码 $exit_code)，已重试 $max_retries 次" >&2
    cat "$tmp_out" >&2
    rm -f "$tmp_out"
    return 1
}

# test vffusion max_parallel and outline
run_test_max_parallel_outline() {
    local file="$1"
    local max_retries=5
    local retry_count=0
    local tmp_out
    tmp_out=$(mktemp)
    local exit_code=0

    echo "############test ($file) max_parallel and outline#############"

    while [ "$retry_count" -lt "$max_retries" ]; do
        > "$tmp_out"

        TRITON_ALWAYS_COMPILE=1 FUSION_MODE=max_parallel ENABLE_VF_OUTLINE=1 \
            python -m pytest -v --tb=short "$file" > "$tmp_out" 2>&1

        exit_code=$?

        if [ "$exit_code" -eq 0 ]; then
            rm -f "$tmp_out"
            return 0
        fi

        retry_count=$((retry_count + 1))
        if [ "$retry_count" -lt "$max_retries" ]; then
            echo "⚠️ 第 $retry_count 次运行失败，重试中... ($file)" >&2
        fi
    done

    echo "❌ $file 运行失败 (退出码 $exit_code)，已重试 $max_retries 次" >&2
    cat "$tmp_out" >&2
    rm -f "$tmp_out"
    return 1
}

# ====================================================
# 单个任务的完整测试逻辑
# normal 与 max_parallel 任意一个失败，整个任务都视为失败。
# ====================================================
run_test_task() {
    local test_file="$1"
    local overall_status=0

    if ! run_test "$test_file"; then
        overall_status=1
    fi

    for file in "${max_parallel_files[@]}"; do
        if [[ "$file" == "$test_file" ]]; then
            if ! run_test_max_parallel "$test_file"; then
                overall_status=1
            fi
            break
        fi
    done

    for file in "${max_parallel_outline_files[@]}"; do
        if [[ "$file" == "$test_file" ]]; then
            if ! run_test_max_parallel_outline "$test_file"; then
                overall_status=1
            fi
            break
        fi
    done

    return "$overall_status"
}

# ====================================================
# 并发执行测试框架
# ====================================================
RESULT_DIR=$(mktemp -d -t test_results_XXXXXX)
echo "并发测试结果目录: $RESULT_DIR"
echo "最大并发数: $MAX_PARALLEL (固定)"
echo "测试 NPU: $ASCEND_RT_VISIBLE_DEVICES"
echo "慢算子标记阈值: 300 秒（仅记录，不终止）"
echo "全脚本超时: 900 秒"

# ====================================================
# 准备待测任务列表（统一管理 oldtest + 算子）
# ====================================================
declare -A TASK_MAP
declare -A TASK_TYPE
TASK_IDS=()
task_count=0
skipped_dirs=()

# 1) oldtest 部分（仅在未指定 test_num 时添加）
if [ -z "$TEST_NUM" ]; then
    old_files=(
        "testcase/oldtest/test_gmm_fwd.py"
        "testcase/oldtest/fa_all.py"
    )

    for file in "${old_files[@]}"; do
        id="oldtest_${task_count}"
        TASK_MAP[$id]="$file"
        TASK_TYPE[$id]="oldtest"
        TASK_IDS+=("$id")
        task_count=$((task_count + 1))
    done
fi

# 2) 算子部分：扫描 testcase 下的目录
cd "$SCRIPT_DIR/testcase" || {
    echo "错误：无法进入 testcase 目录"
    exit 1
}

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

    if [ -n "$TEST_NUM" ] && [ "$dir_num" -ne "$TEST_NUM" ] 2>/dev/null; then
        continue
    fi

    test_file="testcase/$dir/ci.py"

    if [ -f "$test_file" ]; then
        id="op_${dir_num}_${dir}"
        TASK_MAP[$id]="$test_file"
        TASK_TYPE[$id]="operator:${dir}"
        TASK_IDS+=("$id")
        task_count=$((task_count + 1))
    else
        skipped_dirs+=("$dir")
    fi
done

echo "共收集到 $task_count 个测试任务，跳过 ${#skipped_dirs[@]} 个（无 ci.py）"

# ====================================================
# 后台任务管理
# 开启 job control，使每个 (...) & 后台任务拥有独立进程组。
# ====================================================
set -m

launch_test() {
    local id="$1"
    local test_file="$2"
    local log_file="${RESULT_DIR}/${id}.log"
    local status_file="${RESULT_DIR}/${id}.status"

    : > "$log_file"
    echo "RUNNING" > "$status_file"

    (
        # 关闭从父进程继承的仓库锁 FD，防止锁泄露到后台测试进程
        if [ -n "$LOCK_FD" ]; then
            eval "exec $LOCK_FD>&-" 2>/dev/null
        fi

        if run_test_task "$test_file" >> "$log_file" 2>&1; then
            exit 0
        else
            exit 1
        fi
    ) &

    local pid=$!
    TASK_PID[$id]="$pid"
    PID_TASK[$pid]="$id"
    TASK_START[$id]=$(date +%s)
    ACTIVE_PIDS[$pid]=1
    active_jobs=$((active_jobs + 1))
}

# 正常回收已经结束的后台任务，并检查 wait 返回值。
# rc=0 -> PASS
# rc=1 -> FAIL
# 其他退出码 -> ERROR
reap_finished_jobs() {
    local pid id rc now elapsed status_file time_file
    local pids=("${!ACTIVE_PIDS[@]}")

    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        id="${PID_TASK[$pid]}"
        status_file="${RESULT_DIR}/${id}.status"
        time_file="${RESULT_DIR}/${id}.time"

        wait "$pid"
        rc=$?

        now=$(date +%s)
        elapsed=$((now - TASK_START[$id]))
        echo "$elapsed" > "$time_file"

        if [ "$rc" -eq 0 ]; then
            echo "PASS" > "$status_file"
        elif [ "$rc" -eq 1 ]; then
            echo "FAIL" > "$status_file"
        else
            echo "ERROR" > "$status_file"
            echo "❌ 后台测试进程异常退出，退出码: $rc" >> "${RESULT_DIR}/${id}.log"
        fi

        unset 'ACTIVE_PIDS[$pid]'
        unset 'PID_TASK[$pid]'
        active_jobs=$((active_jobs - 1))
    done
}

# 单算子超过 5 分钟不做运行时干预。
# 是否超过 5 分钟根据任务最终记录的耗时在汇总阶段判断。

# 收集结果。
# 已经被调度但状态缺失/RUNNING 都视为 ERROR。
collect_result() {
    local id="$1"
    local status_file="${RESULT_DIR}/${id}.status"
    local st

    if [ ! -f "$status_file" ]; then
        echo "ERROR"
        return
    fi

    st=$(cat "$status_file")

    case "$st" in
        PASS|FAIL|ERROR)
            echo "$st"
            ;;
        RUNNING|*)
            echo "ERROR"
            ;;
    esac
}

# ====================================================
# 并发执行所有任务
# 使用 TASK_IDS 保持 oldtest + 算子的固定、可预测顺序。
# ====================================================
for id in "${TASK_IDS[@]}"; do
    test_file="${TASK_MAP[$id]}"

    while [ "$active_jobs" -ge "$MAX_PARALLEL" ]; do
        reap_finished_jobs

        if [ "$active_jobs" -ge "$MAX_PARALLEL" ]; then
            sleep 0.2
        fi
    done

    echo "启动测试: $id ($test_file)"
    launch_test "$id" "$test_file"
done

# 等待所有后台测试正常完成；单算子超过 5 分钟也不会被终止
while [ "$active_jobs" -gt 0 ]; do
    reap_finished_jobs

    if [ "$active_jobs" -gt 0 ]; then
        sleep 0.2
    fi
done

set +m

# ====================================================
# 在所有任务完成后，先打印失败/异常详情
# ====================================================
echo ""
echo "========== 失败 / 异常任务详细日志 =========="
any_problem=false

for id in "${TASK_IDS[@]}"; do
    result=$(collect_result "$id")

    if [ "$result" = "FAIL" ] || [ "$result" = "ERROR" ]; then
        any_problem=true
        log_file="${RESULT_DIR}/${id}.log"

        echo "--------- 任务: $id [$result] ---------"
        echo "文件: ${TASK_MAP[$id]}"
        echo "--------- 日志（最后 50 行）---------"

        if [ -f "$log_file" ]; then
            tail -n 50 "$log_file"
        else
            echo "未生成日志文件。"
        fi

        echo ""
        echo "完整日志文件: $log_file"
        echo "========================================"
    fi
done

if ! $any_problem; then
    echo "没有失败或异常的任务。"
fi

echo ""

# ====================================================
# 输出每个用例耗时
# 算子超过 5 分钟只标记 SLOW，不改变 PASS/FAIL 状态，也不终止进程。
# ====================================================
echo "========== 每个用例耗时 =========="
for id in "${TASK_IDS[@]}"; do
    time_file="${RESULT_DIR}/${id}.time"
    result=$(collect_result "$id")
    type_info="${TASK_TYPE[$id]}"

    if [ -f "$time_file" ]; then
        elapsed=$(cat "$time_file")
        minutes=$((elapsed / 60))
        seconds=$((elapsed % 60))

        echo -n "$id : ${minutes}m ${seconds}s"

        if [[ "$type_info" == operator:* ]] && [ "$elapsed" -gt 300 ]; then
            echo -n "  ⚠️ SLOW（超�6� 5 分钟，未终止）"
        fi

        if [ "$result" = "ERROR" ]; then
            echo "  ❌ ERROR"
        else
            echo ""
        fi
    else
        echo "$id : 未记录耗时 [$result]"
    fi
done

echo ""

# ====================================================
# 汇总结果统计
# ====================================================
passed_dirs=()
failed_dirs=()
failed_oldtest=()
error_dirs=()
error_oldtest=()
slow_dirs=()

failed_count=0
error_count=0
slow_count=0

for id in "${TASK_IDS[@]}"; do
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
            failed_count=$((failed_count + 1))
            if [[ "$type_info" == "oldtest" ]]; then
                failed_oldtest+=("$test_file")
            else
                dir_name="${type_info#operator:}"
                failed_dirs+=("$dir_name")
            fi
            ;;

        ERROR)
            error_count=$((error_count + 1))
            if [[ "$type_info" == "oldtest" ]]; then
                error_oldtest+=("$test_file")
            else
                dir_name="${type_info#operator:}"
                error_dirs+=("$dir_name")
            fi
            ;;
    esac

    # 慢算子只是性能提示，不影响测试 PASS/FAIL/ERROR 结果。
    time_file="${RESULT_DIR}/${id}.time"
    if [[ "$type_info" == operator:* ]] && [ -f "$time_file" ]; then
        elapsed=$(cat "$time_file")
        if [ "$elapsed" -gt 300 ]; then
            dir_name="${type_info#operator:}"
            slow_dirs+=("$dir_name")
            slow_count=$((slow_count + 1))
        fi
    fi
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

echo "  NPU: $ASCEND_RT_VISIBLE_DEVICES"
echo "  通过: ${#passed_dirs[@]} (算子) + oldtest 通过数未单独统计"
echo "  失败: $failed_count"
echo "  异常: $error_count"
echo "  慢算子: $slow_count (运行超过 5 分钟，仅提示)"
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

if [ ${#error_dirs[@]} -gt 0 ]; then
    echo "异常算子列表:"
    for d in "${error_dirs[@]}"; do
        echo "  - $d"
    done
fi

if [ ${#error_oldtest[@]} -gt 0 ]; then
    echo "异常 oldtest 列表:"
    for f in "${error_oldtest[@]}"; do
        echo "  - $f"
    done
fi

if [ ${#slow_dirs[@]} -gt 0 ]; then
    echo "慢算子列表（运行超过 5 分钟，但未终止）："
    for d in "${slow_dirs[@]}"; do
        echo "  - $d"
    done
fi

if [ ${#skipped_dirs[@]} -gt 0 ]; then
    echo "跳过算子列表:"
    for d in "${skipped_dirs[@]}"; do
        echo "  - $d"
    done
fi

echo "========================================"

echo "------------------------"

# ====================================================
# 日志清理策略
# 有 FAIL / ERROR 时保留 RESULT_DIR；慢算子本身不影响退出码。
# ====================================================
if [ "$failed_count" -eq 0 ] && [ "$error_count" -eq 0 ]; then
    rm -rf "$RESULT_DIR"
    echo "所有测试均通过 ✅"
else
    echo "失败测试数: $failed_count"
    echo "异常测试数: $error_count"
    echo "慢算子数: $slow_count"
    echo "完整日志已保留: $RESULT_DIR"
fi

echo "========Total time: $((SECONDS/60))min $((SECONDS%60))sec ========"

if [ "$failed_count" -eq 0 ] && [ "$error_count" -eq 0 ]; then
    exit 0
else
    exit 1
fi
