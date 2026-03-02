#!/usr/bin/env python3
"""
DLPTI kprof 数据分析脚本
输入: kprof 原始 db 文件
自动调用 export 转换为 metric.db，然后解析

使用前准备:
1. 设置环境:
   source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh
   source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate

2. 采集数据 (生成 kprof.db):
   dlpti_tools kprof --targets cu,tu --data-file ./kprof_out.db -- ./your_app args

3. 运行分析:
   python3 analyze_kprof.py ./kprof_out.db
"""

import sqlite3
import subprocess
import sys


def export_to_metric(db_path):
    """调用 dlpti_tools export 转换为 metric.db"""
    metric_path = db_path.replace('.db', '.metric.db')

    print("正在 export 为 metric.db...")
    cmd = [
        'dlpti_tools', 'export',
        '--format', 'metric',
        '--output', metric_path,
        '--skip-obfusion',
        db_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Export 失败: %s" % result.stderr)
        return None

    print("生成: %s" % metric_path)
    return metric_path


def analyze_kprof(metric_path):
    """解析 kprof metric.db 文件"""
    print("\n" + "=" * 70)
    print("kprof 分析: %s" % metric_path)
    print("=" * 70)

    conn = sqlite3.connect(metric_path)
    cursor = conn.cursor()

    # Profiler 信息
    cursor.execute("SELECT name, version FROM profiler_info")
    row = cursor.fetchone()
    print("\n[Profiler]")
    print("  %s v%s" % (row[0], row[1]))

    # Kernel 信息
    cursor.execute("SELECT kprof_target_name FROM kprof_target_info")
    row = cursor.fetchone()
    print("\n[Kernel]")
    print("  名称: %s" % row[0])

    # Grid/Block 配置
    cursor.execute("SELECT kprof_target_params FROM kprof_target_info")
    row = cursor.fetchone()
    if row[0]:
        import json
        params = json.loads(row[0])
        grid = params.get('grid_size', [0, 0, 0])
        block = params.get('block_size', [0, 0, 0])
        print("  grid: (%d, %d, %d), block: (%d, %d, %d)" % (
            grid[0], grid[1], grid[2], block[0], block[1], block[2]))

    # Kernel 执行时间
    print("\n[Kernel 执行时间 ce__kernel_elapsed_cycles]")
    cursor.execute("""
        SELECT invoke_id, AVG(value), MIN(value), MAX(value)
        FROM kprof_metric
        WHERE metric_name = 'ce__kernel_elapsed_cycles'
        GROUP BY invoke_id
        ORDER BY invoke_id
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print("  invoke %d: avg=%.0f, min=%.0f, max=%.0f cycles" % row)

    # 总体统计
    cursor.execute("SELECT AVG(value), MIN(value), MAX(value), COUNT(*) FROM kprof_metric WHERE metric_name = 'ce__kernel_elapsed_cycles'")
    row = cursor.fetchone()
    print("\n[总体统计]")
    print("  平均: %.1f cycles" % row[0])
    print("  最小: %.1f cycles" % row[1])
    print("  最大: %.1f cycles" % row[2])
    print("  次数: %d" % row[3])
    print("  换算 (1GHz): %.3f us" % (row[0] / 1000))

    # 其他关键指标
    print("\n[其他关键指标]")
    metrics = [
        ('fe__dma_elapsed_cycles', 'DMA'),
        ('fe__gtd_elapsed_cycles', 'GTD'),
        ('fe__flush_elapsed_cycles', 'Flush'),
        ('ce__active_cycles', 'CE Active'),
    ]
    print("  指标 | 平均值 (cycles)")
    print("  -----|----------------")
    for metric_name, desc in metrics:
        cursor.execute("SELECT AVG(value) FROM kprof_metric WHERE metric_name = ?", (metric_name,))
        row = cursor.fetchone()
        if row[0]:
            print("  %s | %.1f" % (desc, row[0]))

    conn.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 analyze_kprof.py <kprof.db>")
        print("\n前置步骤:")
        print("  1. source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh")
        print("  2. source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate")
        print("  3. dlpti_tools kprof --targets cu,tu --data-file ./kprof_out.db -- ./your_app args")
        sys.exit(1)

    db_path = sys.argv[1]

    # 转换为 metric.db
    metric_path = export_to_metric(db_path)
    if not metric_path:
        sys.exit(1)

    # 解析
    analyze_kprof(metric_path)
