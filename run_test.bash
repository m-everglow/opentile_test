#!/bin/bash
# 用法: ./run_test.bash [pr=<PR编号>] [test_num=<算子编号>]
# 功能：编译并运行测试。若未指定 pr，则使用 main 分支；
#       若不指定 test_num，则运行 oldtest + 所有算子；
#       若指定 test_num，则仅运行对应编号的算子（如 test_num=2 -> 跑 02_xxx 或 2_xxx）

echo "========== 脚本开始 =========="

# ---- 获取脚本所在目录（绝对路径） ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "脚本目录: $SCRIPT_DIR"

source /data/set_proxy.bash
source /data/setenv.bash
rm -rf /tmp/torchinductor_root/*
rm -rf ~/.triton/dump
rm -rf ~/.triton/cache

# ---- 全局锁 ----
LOCKFILE="/tmp/OpenTileAS_build.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "错误：另一个脚本实例正在运行（可能由其他用户或不同 PR 启动），请等待其完成后再试。"
    exit 1
fi

# ====================================================
# 参数解析：支持 pr=xxx 和 test_num=xx
# ====================================================
MODE="main"
PR_NUM=""
BRANCH_NAME=""
TEST_NUM=""          # 若指定，只运行该编号的算子

for arg in "$@"; do
    if [[ "$arg" =~ ^pr=([0-9]+)$ ]]; then
        PR_NUM="${BASH_REMATCH[1]}"
        MODE="pr"
    elif [[ "$arg" =~ ^test_num=([0-9]+)$ ]]; then
        TEST_NUM="${BASH_REMATCH[1]}"
    fi
done

if [ -n "$PR_NUM" ]; then
    BRANCH_NAME="pr_${PR_NUM}"
fi

# ---- 清理函数（无论成功或失败都会执行） ----
cleanup() {
    echo "正在进行清理..."
    cd "$SCRIPT_DIR/OpenTileAS" 2>/dev/null || return
    git checkout main 2>/dev/null || true
    git reset --hard origin/main 2>/dev/null || true
    if [ -n "$BRANCH_NAME" ]; then
        git branch -D "${BRANCH_NAME}" 2>/dev/null || true
    fi
    cd "$SCRIPT_DIR" 2>/dev/null || return
    echo "清理完成"
}
trap cleanup EXIT

# ---- 进入仓库目录 ----
cd "$SCRIPT_DIR/OpenTileAS" || { echo "错误：无法进入 $SCRIPT_DIR/OpenTileAS 目录"; exit 1; }

echo "正在切换到 main 分支并强制拉取最新代码..."
git checkout main || exit 1
git fetch origin main || exit 1
git reset --hard origin/main || exit 1

if [ "$MODE" = "pr" ]; then
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
else
    echo "未指定 PR 编号，使用最新 main 分支"
fi

echo "正在清理旧的 build 目录..."
rm -rf build
mkdir build
cd build || exit 1

echo "正在执行 cmake ..."
cmake .. || exit 1

echo "正在执行 make -j48 ..."
make -j48 || exit 1

# ---- 返回脚本目录 ----
cd "$SCRIPT_DIR" || exit 1
export PATH=/data/y00939135/test/OpenTileAS/build/bin:$PATH
echo "========== 编译完成 =========="

# ====================================================
# 定义测试运行函数（参数为相对于 SCRIPT_DIR 的文件路径）
# ====================================================
run_test() {
    local file="$1"
    echo "------------------------"
    echo "运行: $file"
    local tmp_out=$(mktemp)
    if [[ "$file" == *"fa.py" ]]; then
        TRITON_ALWAYS_COMPILE=1 python "$file" > "$tmp_out" 2>&1
    else
        TRITON_ALWAYS_COMPILE=1 pytest -v --tb=short "$file" > "$tmp_out" 2>&1
    fi
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ $file 通过"
        rm -f "$tmp_out"
        return 0
    else
        echo "❌ $file 运行失败 (退出码 $exit_code)"
        if [[ "$file" != *"fa.py" ]]; then
            local failed_tests=$(grep -E "FAILED|ERROR" "$tmp_out" | grep -oE "\S+::\S+" | sort -u)
            if [ -n "$failed_tests" ]; then
                echo "失败的用例:"
                echo "$failed_tests"
            else
                echo "无法解析失败用例，请查看完整日志:"
                cat "$tmp_out"
            fi
        else
            echo "完整日志:"
            cat "$tmp_out"
        fi
        rm -f "$tmp_out"
        return 1
    fi
}

# ====================================================
# 第一阶段：执行 oldtest（仅当未指定 test_num 时）
# ====================================================
failed_files=()      # 用于记录失败的测试文件（包括oldtest和算子）
failed_count=0

if [ -z "$TEST_NUM" ]; then
    echo "========== 第一阶段：执行 oldtest 目录下的原有测试（硬编码列表） =========="

    # 硬编码列表，路径前缀为 testcase/oldtest/
    old_files=(
        "testcase/oldtest/test_gmm_fwd.py"
        "testcase/oldtest/test_rmsnorm.py"
        "testcase/oldtest/fa.py"
        "testcase/oldtest/_fused_add_layernorm_fwd_kernel.py"
        "testcase/oldtest/_fused_add_rmsnorm_fwd_kernel.py"
        "testcase/oldtest/test_fused_matmul_bwd_x.py"
        "testcase/oldtest/test_silu_forward_backward_diff.py"
        "testcase/oldtest/test_fused_ce_forward_backward_diff.py"
    )

    for file in "${old_files[@]}"; do
        if ! run_test "$file"; then
            failed_files+=("$file")
            ((failed_count++))
        fi
    done
else
    echo "========== 指定 test_num=${TEST_NUM}，跳过 oldtest 阶段 =========="
fi

# ====================================================
# 第二阶段：执行算子（所有或单个）
# ====================================================
if [ -z "$TEST_NUM" ]; then
    echo "========== 第二阶段：执行所有算子（ci.py） =========="
else
    echo "========== 第二阶段：仅执行编号 ${TEST_NUM} 的算子 =========="
fi

cd "$SCRIPT_DIR/testcase" || { echo "错误：无法进入 testcase 目录"; exit 1; }
shopt -s nullglob
test_dirs=( [0-9]*_*/ )
shopt -u nullglob

if [ ${#test_dirs[@]} -eq 0 ]; then
    echo "错误：testcase 下没有找到算子目录（格式如 01_xxx/）"
    exit 1
fi

# 按数字顺序排序
IFS=$'\n' test_dirs=($(sort <<<"${test_dirs[*]}"))
unset IFS

cd "$SCRIPT_DIR" || exit 1

# 记录通过/失败/跳过的算子目录（仅名称，不含路径）
passed_dirs=()
failed_dirs=()
skipped_dirs=()

# 筛选目录：如果指定了 TEST_NUM，只处理数字部分与之相等的目录
selected_dirs=()
for dir in "${test_dirs[@]}"; do
    dir="${dir%/}"
    # 提取下划线前的数字
    dir_num="${dir%%_*}"
    if [ -z "$TEST_NUM" ] || [ "$dir_num" -eq "$TEST_NUM" ] 2>/dev/null; then
        selected_dirs+=("$dir")
    fi
done

if [ -n "$TEST_NUM" ] && [ ${#selected_dirs[@]} -eq 0 ]; then
    echo "错误：未找到编号为 ${TEST_NUM} 的算子目录"
    exit 1
fi

for dir in "${selected_dirs[@]}"; do
    echo "------------------------"
    echo "运行算子目录: $dir"

    # 检查是否存在 ci.py
    test_file="testcase/$dir/ci.py"
    if [ ! -f "$test_file" ]; then
        echo "⚠️ 目录 $dir 中没有找到 ci.py，跳过"
        skipped_dirs+=("$dir")
        continue
    fi

    if run_test "$test_file"; then
        passed_dirs+=("$dir")
    else
        failed_dirs+=("$dir")
        failed_files+=("$dir")
        ((failed_count++))
    fi
done

# ====================================================
# 输出统计信息
# ====================================================
echo "========================================"
if [ -n "$TEST_NUM" ]; then
    echo "仅运行算子编号 ${TEST_NUM} 的测试结果："
else
    echo "所有算子测试统计："
fi
echo "  通过: ${#passed_dirs[@]}"
echo "  失败: ${#failed_dirs[@]}"
echo "  跳过: ${#skipped_dirs[@]} (未找到 ci.py)"
if [ ${#failed_dirs[@]} -gt 0 ]; then
    echo "失败算子列表:"
    for d in "${failed_dirs[@]}"; do
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

# ====================================================
# 汇总最终结果（整体）
# ====================================================
echo "------------------------"
if [ $failed_count -eq 0 ]; then
    if [ -n "$TEST_NUM" ]; then
        echo "该算子测试通过 ✅"
    else
        echo "所有测试（oldtest + 算子）均通过 ✅"
    fi
else
    echo "失败的测试总数: $failed_count"
    echo "失败的测试项:"
    for f in "${failed_files[@]}"; do
        echo "  - $f"
    done
fi

if [ $failed_count -eq 0 ]; then
    exit 0
else
    exit 1
fi
