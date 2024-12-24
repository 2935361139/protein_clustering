# convert_tsv_to_labels.py

import os
import csv

def convert_tsv_to_labels(input_tsv, output_tsv, target_iprs):
    """
    从输入TSV文件中生成多标签标签文件。
    每个seq_id对应多个标签，表示是否包含目标IPR编号。

    参数：
    - input_tsv: 输入的TSV文件路径。
    - output_tsv: 输出的标签文件路径。
    - target_iprs: 目标IPR编号列表，例如 ["IPR049892", "IPR018392"]。
    """
    if not os.path.exists(input_tsv):
        print(f"Input TSV file {input_tsv} does not exist.")
        return

    # 初始化字典，键为seq_id，值为标签列表
    seq_to_labels = {}

    with open(input_tsv, 'r', encoding='utf-8') as infile:
        for line in infile:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            parts = line_stripped.split('\t')
            # 假设seq_id在第0列
            seq_id = parts[0].strip()

            # 初始化标签列表为0
            if seq_id not in seq_to_labels:
                seq_to_labels[seq_id] = [0] * len(target_iprs)

            # 检查整行是否包含任一目标IPR
            for idx, ipr in enumerate(target_iprs):
                if ipr in parts:
                    seq_to_labels[seq_id][idx] = 1

    # 写入输出文件
    with open(output_tsv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        # 写入表头
        header = ['seq_id'] + target_iprs
        writer.writerow(header)
        # 写入每个seq_id及其标签
        for seq_id, labels in seq_to_labels.items():
            writer.writerow([seq_id] + labels)

    # 打印统计信息
    total_seqs = len(seq_to_labels)
    print(f"Labels saved to {output_tsv}. Total sequences: {total_seqs}")

if __name__ == "__main__":
    # 直接在此处填写输入输出路径和目标IPR编号列表
    input_tsv = "D:/bjfu/GNN-transformer/protein_cluster/data/Pfam-zhushi/0.tsv"
    output_tsv = "D:/bjfu/GNN-transformer/protein_cluster/data/Pfam-zhushi/output_labels.tsv"
    # 填写您选择的目标IPR编号列表
    target_iprs = ["IPR000026","IPR000070","IPR000254","IPR000334","IPR000566","IPR000675",
                   "IPR000743","IPR000757","IPR001002","IPR001087","IPR001137","IPR001179",
                   "IPR001254","IPR001283","IPR001305","IPR001314","IPR001580","IPR001621",
                   "IPR001623","IPR001667","IPR001722","IPR002016","IPR002022","IPR002044",
                   "IPR002048","IPR002130","IPR002889","IPR002939","IPR003172","IPR004097",
                   "IPR004302","IPR004352","IPR004843","IPR004898","IPR005103","IPR005788",
                   "IPR006047","IPR006626","IPR007074","IPR007484","IPR007567","IPR007934",
                   "IPR008030","IPR008119","IPR008427","IPR008701","IPR008971","IPR008972",
                   "IPR009003","IPR009011","IPR009033","IPR009038","IPR009644","IPR010126",
                   "IPR010255","IPR010636","IPR010829","IPR010895","IPR011050","IPR011058",
                   "IPR011118","IPR011679","IPR011992","IPR012334","IPR012358","IPR012674",
                   "IPR013319","IPR013320","IPR013766","IPR013777","IPR013780","IPR013783",
                   "IPR013784","IPR013785","IPR014044","IPR014756","IPR015141","IPR015289",
                   "IPR015720","IPR016191","IPR017853","IPR017937","IPR018114","IPR018124",
                   "IPR018208","IPR018244","IPR018247","IPR018253","IPR018392","IPR018466",
                   "IPR018535","IPR018803","IPR018939","IPR019794","IPR020892","IPR021054",
                   "IPR021476","IPR022271","IPR024079","IPR024461","IPR025649","IPR028146",
                   "IPR029000","IPR029052","IPR029058","IPR029226","IPR032710","IPR033116",
                   "IPR033119","IPR033123","IPR033131","IPR033917","IPR034543","IPR034836",
                   "IPR035940","IPR035971","IPR036195","IPR036249","IPR036291","IPR036356",
                   "IPR036410","IPR036444","IPR036514","IPR036598","IPR036607","IPR036673",
                   "IPR036686","IPR036779","IPR036861","IPR036869","IPR036908","IPR037019",
                   "IPR037045","IPR037401","IPR038222","IPR038763","IPR038843","IPR038903",
                   "IPR038964","IPR039448","IPR039513","IPR039670","IPR039794","IPR040250",
                   "IPR043504","IPR044609","IPR044713","IPR044831","IPR044865","IPR045032",
                   "IPR045175","IPR046357","IPR046530","IPR046936","IPR047202","IPR048269",
                   "IPR049892","IPR050434","IPR050546","IPR050955","IPR051063","IPR051164",
                   "IPR051386","IPR051425","IPR051550","IPR051589","IPR051694","IPR051735",
                   "IPR052052","IPR052063","IPR052210","IPR052282","IPR052288","IPR052471",
                   "IPR052606","IPR052820","IPR052953","IPR052982","IPR053216","IPR053868",
                   "IPR056124"
]  # 示例，请根据实际需求修改

    convert_tsv_to_labels(input_tsv, output_tsv, target_iprs)
