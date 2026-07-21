#!/bin/bash
# 用法: ./script.sh <PR编号>
# 功能：拉取指定 PR 的代码，切换至 pr_<编号> 分支，rebase 到最新 main，编译，
#       运行三个 pytest 测试，统计失败用例，最后切回 main（不删除分支）

echo "========== 脚本开始 =========="

if [ $# -lt 1 ]; then
    echo "错误：缺少参数（PR 编号）"
    echo "用法: $0 <PR编号>"
    exit 1
fi

PR_NUM="$1"
BRANCH_NAME="pr_${PR_NUM}"

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

# ---- 进入仓库目录 ----
cd "$SCRIPT_DIR/OpenTileAS" || { echo "错误：无法进入 $SCRIPT_DIR/OpenTileAS 目录"; exit 1; }
git checkout main 
# ---- 清理本地同名分支（确保 fetch 能正常创建/更新） ----
echo "正在删除可能存在的本地分支 ${BRANCH_NAME} ..."
git branch -D "${BRANCH_NAME}" 2>/dev/null || true

echo "正在从远程拉取最新的 main 分支..."
git fetch origin main || exit 1

echo "正在从远程拉取 PR #${PR_NUM} 的代码..."
git fetch https://gitcode.com/OpenTileIR/OpenTileAS.git "+refs/merge-requests/${PR_NUM}/head:${BRANCH_NAME}" || exit 1

echo "正在切换到 ${BRANCH_NAME} 分支..."
git checkout "${BRANCH_NAME}" || exit 1

echo "正在将 PR 分支 rebase 到最新的 origin/main 上..."

git rebase --abort
git rebase origin/main || {
    echo "错误：rebase 失败，可能存在冲突，请手动解决"
    exit 1
}

echo "正在清理旧的 build 目录..."
rm -rf build
mkdir build
cd build || exit 1

echo "正在执行 cmake ..."
cmake .. || exit 1

echo "正在执行 make -j6 ..."
make -j6 || exit 1

# ---- 返回脚本目录 ----
cd "$SCRIPT_DIR" || exit 1
pwd
echo "========== 编译完成 =========="

export PATH=/data/y00939135/test/OpenTileAS/build/bin:$PATH
# source /data/setenv.bash

# ---- 运行测试（测试文件在脚本目录） ----
run_test() {
    local file=$1
    echo "------------------------"
    echo "运行 pytest: $file"
    local tmp_out=$(mktemp)
    TRITON_ALWAYS_COMPILE=1 pytest -v --tb=short "$file" > "$tmp_out" 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ $file 通过"
        rm -f "$tmp_out"
        return 0
    else
        echo "❌ $file 运行失败 (退出码 $exit_code)"
        local failed_tests=$(grep -E "FAILED|ERROR" "$tmp_out" | grep -oE "\S+::\S+" | sort -u)
        if [ -n "$failed_tests" ]; then
            echo "失败的用例:"
            echo "$failed_tests"
        else
            echo "无法解析失败用例，请查看完整日志:"
            cat "$tmp_out"
        fi
        rm -f "$tmp_out"
        return 1
    fi
}

failed_files=()
failed_count=0
for file in "test_gmm_fwd.py" "test_rmsnorm.py" "fa.py"; do
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

# ---- 切回 main 并重置到远程最新（不删除分支） ----
echo "正在切回 main 分支并重置到 origin/main ..."
cd "$SCRIPT_DIR/OpenTileAS" || { echo "错误：无法进入 $SCRIPT_DIR/OpenTileAS 目录"; exit 1; }
git checkout main || exit 1
git reset --hard origin/main || exit 1

# 返回脚本目录（可选）
cd "$SCRIPT_DIR" || exit 1

# 最终根据测试结果退出
if [ $failed_count -eq 0 ]; then
    exit 0
else
    exit 1
fi
