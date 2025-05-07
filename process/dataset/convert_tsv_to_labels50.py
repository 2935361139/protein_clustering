# convert_tsv_to_labels.py

import pandas as pd
import os


def convert_tsv_to_top50(tsv_input, tsv_output, top_n=50):
    """
    将原始多标签 TSV 文件转换为仅包含前 N 个最常见的 IPR 列。

    参数：
    - tsv_input: 原始多标签 TSV 文件路径。
    - tsv_output: 转换后 TSV 文件的保存路径。
    - top_n: 选择最常见的前 N 个 IPR。
    """
    if not os.path.exists(tsv_input):
        print(f"输入文件 {tsv_input} 不存在。")
        return

    # 读取 TSV 文件
    df = pd.read_csv(tsv_input, sep='\t')

    # 确保 'seq_id' 列存在
    if 'seq_id' not in df.columns:
        print("TSV 文件中缺少 'seq_id' 列。")
        return

    # 统计每个 IPR 列中 1 的数量
    ipr_columns = [col for col in df.columns if col != 'seq_id']
    ipr_counts = df[ipr_columns].sum().sort_values(ascending=False)

    # 选择前 top_n 个 IPR
    top_iprs = ipr_counts[ipr_counts > 0].head(top_n).index.tolist()

    if len(top_iprs) < top_n:
        print(f"只找到 {len(top_iprs)} 个有 '1' 的 IPR。")

    # 选择 'seq_id' 和前 top_n 个 IPR 列
    df_top = df[['seq_id'] + top_iprs]

    # 保存转换后的 TSV 文件
    df_top.to_csv(tsv_output, sep='\t', index=False)
    print(f"转换完成，保存到 {tsv_output}，包含 {len(top_iprs)} 个 IPR 列。")


if __name__ == "__main__":
    # 设置输入和输出路径
    tsv_input = ""
    tsv_output = ""

    # 执行转换
    convert_tsv_to_top50(tsv_input, tsv_output, top_n=10)
