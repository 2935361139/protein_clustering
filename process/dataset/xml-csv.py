# xml-csv.py
import xml.etree.ElementTree as ET
import csv
from Bio import SeqIO


def load_vdia_ids(fasta_file):
    """
    从FASTA文件中加载V.dahliae的序列ID到集合。
    假设FASTA文件中序列ID与PDB文件名一致，如 VDAG_00063。
    """
    vdia_ids = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        vdia_ids.add(record.id)
    return vdia_ids


def blast_xml_to_csv(xml_file, csv_file, vdia_ids):
    """
    将BLAST XML结果解析为CSV格式的(seq1_id, seq2_id, e_value)。
    使用<Iteration_query-def>作为seq1_id（应为VDAG_xxx格式）
    使用<Hit_def>作为seq2_id（也应为VDAG_xxx格式）
    只保留seq2_id在V.dahliae序列ID集合中的记录，且排除自匹配。
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    fieldnames = ['seq1_id', 'seq2_id', 'e_value']

    with open(csv_file, 'w', newline='', encoding='utf-8') as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历所有 Iterations
        for iteration in root.findall('./BlastOutput_iterations/Iteration'):
            # 获取 seq1_id
            query_def_elem = iteration.find('Iteration_query-def')
            if query_def_elem is None:
                continue
            seq1_id = query_def_elem.text.strip()

            # 遍历所有 Hits
            hits = iteration.findall('Iteration_hits/Hit')
            for hit in hits:
                hit_def_elem = hit.find('Hit_def')
                if hit_def_elem is None:
                    continue
                raw_seq2_id = hit_def_elem.text.strip()

                # 过滤：仅保留 V.dahliae 序列且不等于 seq1_id（排除自匹配）
                if raw_seq2_id not in vdia_ids or raw_seq2_id == seq1_id:
                    continue

                # 遍历 HSPs
                hsps = hit.findall('Hit_hsps/Hsp')
                for hsp in hsps:
                    evalue_elem = hsp.find('Hsp_evalue')
                    if evalue_elem is not None:
                        e_value = evalue_elem.text.strip()
                        writer.writerow({
                            'seq1_id': seq1_id,
                            'seq2_id': raw_seq2_id,
                            'e_value': e_value
                        })


if __name__ == "__main__":
    # 输入文件路径（请根据实际情况修改）
    xml_input_path = "D:/bjfu/GNN-transformer/protein_cluster/data/csv/v1.xml"
    csv_output_path = "D:/bjfu/GNN-transformer/protein_cluster/data/csv/v3.csv"
    fasta_file = "D:/bjfu/GNN-transformer/protein_cluster/data/fasta/nrAF_PDB_train_sequences.fasta"

    # 加载V.dahliae序列ID
    vdia_ids = load_vdia_ids(fasta_file)

    # 转换XML到CSV
    blast_xml_to_csv(xml_input_path, csv_output_path, vdia_ids)
    print(f"数据已成功转换为CSV格式: {csv_output_path}")
