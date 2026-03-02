#!/usr/bin/env python3
"""
DLPTI capture 数据分析脚本
输入: capture 原始 db 文件
自动调用 export 转换为 perfetto.json，然后解析

使用前准备:
1. 设置环境:
   source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh
   source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate

2. 采集数据 (生成 capture.db):
   dlpti_tools capture --data-file ./capture_out.db -- ./your_app args

3. 运行分析:
   python3 analyze_capture.py ./capture_out.db
"""

import sqlite3
import subprocess
import sys
import json


def export_to_perfetto(db_path):
    """调用 dlpti_tools export 转换为 perfetto JSON"""
    json_path = db_path.replace('.db', '.perfetto.json')

    print("正在 export 为 perfetto JSON...")
    cmd = ['dlpti_tools', 'export', '--format', 'perfetto-json', db_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Export 失败: %s" % result.stderr)
        return None

    print("生成: %s" % json_path)
    return json_path


def analyze_capture(db_path, json_path):
    """分析 capture db 和 perfetto JSON"""
    print("\n" + "=" * 70)
    print("capture 分析: %s" % db_path)
    print("=" * 70)

    # 1. 解析 db 基本信息
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name, version FROM profiler_info")
    row = cursor.fetchone()
    print("\n[Profiler]")
    print("  %s v%s" % (row[0], row[1]))

    cursor.execute("SELECT start_time, end_time FROM target")
    row = cursor.fetchone()
    if row[0] and row[1]:
        duration_us = (row[1] - row[0]) / 1000
        print("\n[时间范围]")
        print("  总时长: %.2f ms" % (duration_us / 1000))

    kind_names = {1: "FUNCTION", 2: "MEMCPY", 17: "KERNEL", 18: "MEMSET"}
    cursor.execute("SELECT kind, COUNT(*) FROM activity GROUP BY kind ORDER BY COUNT(*) DESC")
    print("\n[Activity 分布]")
    for row in cursor.fetchall():
        name = kind_names.get(row[0], "kind_%d" % row[0])
        print("  %s: %d" % (name, row[1]))

    conn.close()

    # 2. 解析 perfetto JSON
    print("\n[Perfetto JSON 分析]")
    with open(json_path) as f:
        data = json.load(f)

    events = data['traceEvents']

    # Kernel 执行时间 (CU COMMAND)
    cu_events = [e for e in events if e.get('name') == 'CU' and e.get('cat') == 'COMMAND']
    if cu_events:
        durs = [e['dur'] for e in cu_events if e.get('dur')]
        if durs:
            kernel_name = cu_events[0]['args'].get('kernel_name', 'unknown')
            print("\n[Kernel 执行时间]")
            print("  名称: %s" % kernel_name)
            print("  平均: %.3f us" % (sum(durs) / len(durs)))
            print("  最小: %.3f us" % min(durs))
            print("  最大: %.3f us" % max(durs))
            print("  次数: %d" % len(durs))

    # API 调用时间
    print("\n[API 调用时间]")
    apis = [
        ('cuLaunchKernel', 'DRIVER_API'),
        ('cuEventRecord', 'DRIVER_API'),
        ('cudaLaunchKernel', 'RUNTIME_API'),
    ]
    print("  API | 平均 (us) | 最小 | 最大 | 次数")
    print("  ----|-----------|------|------|------")
    for api_name, cat in apis:
        api_events = [e for e in events if e.get('name') == api_name and e.get('cat') == cat]
        durs = [e['dur'] for e in api_events if e.get('dur')]
        if durs:
            durs_sorted = sorted(durs)
            durs_filtered = durs_sorted[:-1] if len(durs_sorted) > 1 else durs_sorted
            print("  %s | %.3f | %.3f | %.3f | %d" % (
                api_name, sum(durs_filtered) / len(durs_filtered),
                min(durs_filtered), max(durs_filtered), len(durs_filtered)
            ))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 analyze_capture.py <capture.db>")
        print("\n前置步骤:")
        print("  1. source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh")
        print("  2. source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate")
        print("  3. dlpti_tools capture --data-file ./capture_out.db -- ./your_app args")
        sys.exit(1)

    db_path = sys.argv[1]

    # 转换为 perfetto JSON
    json_path = export_to_perfetto(db_path)
    if not json_path:
        sys.exit(1)

    # 解析
    analyze_capture(db_path, json_path)
