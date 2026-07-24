export PATH=/data/y00939135/test/OpenTileAS/build/bin:$PATH

# ---- 运行测试 ----
run_test() {
    local file=$1
    echo "------------------------"
    echo "运行: $file"
    local tmp_out=$(mktemp)
    if [[ "$file" == "fa.py" ]]; then
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
        if [[ "$file" == "fa.py" ]]; then
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
for file in "fa.py"; do
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

# ---- 切回 main 并重置到远程最新 ----
echo "正在切回 main 分支并重置到 origin/main ..."
cd "$SCRIPT_DIR/OpenTileAS" || { echo "错误：无法进入 $SCRIPT_DIR/OpenTileAS 目录"; exit 1; }
git checkout main || exit 1
git reset --hard origin/main || exit 1

cd "$SCRIPT_DIR" || exit 1

# 最终根据测试结果退出
if [ $failed_count -eq 0 ]; then
    exit 0
else
    exit 1
fi
