#!/bin/bash
# 用法: ./script.sh [<PR编号>]
# 功能：带 PR 编号则拉取该 PR 代码并 rebase main，不带则直接拉取最新 main，
#       然后编译并运行测试，最后自动清理

echo "========== 脚本开始 =========="

# ---- 获取脚本所在目录（绝对路径） ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "脚本目录: $SCRIPT_DIR"

source /data/set_proxy.bash
source /data/setenv.bash

# ---- 全局锁 ----
LOCKFILE="/tmp/OpenTileAS_build.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "错误：另一个脚本实例正在运行（可能由其他用户或不同 PR 启动），请等待其完成后再试。"
    exit 1
fi

# ---- 参数解析 ----
MODE="main"
PR_NUM=""
BRANCH_NAME=""

if [ $# -ge 1 ]; then
    MODE="pr"
    PR_NUM="$1"
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
    # ---- 清理本地同名分支（确保 fetch 能正常创建/更新） ----
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
pwd
echo "========== 编译完成 =========="

export PATH=/data/y00939135/test/OpenTileAS/build/bin:$PATH

# ---- 运行测试 ----
run_test() {
    local file=$1
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
        if [[ "$file" == *"fa.py" ]]; then
            echo "完整日志:"
            cat "$tmp_out"
        else
            local failed_tests=$(grep -E "FAILED|ERROR" "$tmp_out" | grep -oE "\S+::\S+" | sort -u)
            if [ -n "$failed_tests" ]; then
                echo "失败的用例:"
                echo "$failed_tests"
            else
                echo "无法解析失败用例，请查看完整日志:"
                cat "$tmp_out"
            fi
        fi
        rm -f "$tmp_out"
        return 1
    fi
}

failed_files=()
failed_count=0
for file in  "testcase/test_gmm_fwd.py" "testcase/test_rmsnorm.py" "testcase/fa.py" "testcase/gelu.py" "testcase/silu.py" "testcase/swiglu.py" "testcase/paged_decode.py" "testcase/_fused_add_layernorm_fwd_kernel.py" "testcase/_fused_add_rmsnorm_fwd_kernel.py"; do
    if ! run_test "$file"; then
        failed_files+=("$file")
        ((failed_count++))
    fi
done

echo "------------------------"
if [ $failed_count -eq 0 ]; then
    echo "所有测试文件均通过 ✅"
else
    echo "错误数量: $failed_count"
    echo "错误的脚本:"
    for f in "${failed_files[@]}"; do
        echo "  - $f"
    done
fi

if [ $failed_count -eq 0 ]; then
    exit 0
else
    exit 1
fi
