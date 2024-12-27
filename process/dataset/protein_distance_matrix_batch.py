# protein_distance_matrix_batch.py
import os
import numpy as np
from Bio import PDB
from Bio.PDB import PPBuilder
from Bio import pairwise2
import pandas as pd
import math
from Bio import SeqIO


############################################
# Step 1: Load V.dahliae sequences
############################################

def load_vdia_sequences(fasta_file):
    """
    加载V.dahliae的序列到字典中，映射 {seq_id: sequence}

    Args:
        fasta_file (str): V.dahliae FASTA文件路径
    Returns:
        dict: {seq_id: sequence}
    """
    fasta_to_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        fasta_to_dict[record.id] = str(record.seq)
    return fasta_to_dict


############################################
# Step 2: Parse BLAST CSV
############################################

def read_blast_results(blast_result_file):
    """
    读取BLAST结果CSV文件，假设列为: seq1_id, seq2_id, e_value
    """
    df = pd.read_csv(blast_result_file)
    return df


def get_best_match_sequence_id(current_seq_id, blast_df, vdia_ids):
    """
    给定当前PDB序列ID，从BLAST结果中找到与current_seq_id匹配的V.dahliae序列ID及最优E值。
    如果找不到匹配，返回None, None。

    Args:
        current_seq_id (str): 当前PDB序列ID
        blast_df (pd.DataFrame): BLAST结果的DataFrame
        vdia_ids (set): V.dahliae序列ID集合
    Returns:
        tuple: (match_id, e_value) or (None, None)
    """
    # 找出所有与current_seq_id匹配的行
    mask = (blast_df['seq1_id'] == current_seq_id) | (blast_df['seq2_id'] == current_seq_id)
    sub_df = blast_df[mask]

    if sub_df.empty:
        return None, None

    # 找到 E 值最小的一行
    best_row = sub_df.loc[sub_df['e_value'].idxmin()]

    # 确定匹配序列 id
    if best_row['seq1_id'] == current_seq_id:
        match_id = best_row['seq2_id']
    else:
        match_id = best_row['seq1_id']

    return match_id, best_row['e_value']


def evalue_to_similarity(e_value):
    """
    将E值映射为相似度评分。例如使用 -log10(E+1e-10) 归一化到0~1区间。
    这里是示例逻辑，您可根据生物学意义进一步优化。
    """
    try:
        e_value = float(e_value)
    except ValueError:
        return 0.0
    score = -math.log10(e_value + 1e-10)
    # 假设score最多达到10（很小的e_value）。如果超过10则截断。
    score = min(score, 10.0)
    # 归一化到0~1之间
    normalized = score / 10.0
    return normalized


############################################
# Step 3: Extract PDB data and calculate distance matrices
############################################

def extract_ca_coordinates(pdb_file):
    """
    从PDB文件中提取Cα原子坐标。
    返回一个坐标列表和对应的残基信息（用于后续映射）。
    """
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
    except Exception as e:
        print(f"Error parsing PDB file {pdb_file}: {e}")
        return np.array([]), []
    model = structure[0]
    ca_atoms = []
    residue_ids = []
    for chain in model:
        for residue in chain:
            if 'CA' in residue:
                ca_atoms.append(residue['CA'].coord)
                residue_ids.append((chain.id, residue.get_id()[1], residue.get_resname()))
    return np.array(ca_atoms), residue_ids


def get_pdb_sequence(pdb_file, chain_id='A'):
    """
    从PDB文件中提取指定链的序列。
    """
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
    except Exception as e:
        print(f"Error parsing PDB file {pdb_file}: {e}")
        return ""
    model = structure[0]
    if chain_id not in model:
        print(f"Chain {chain_id} not found in {pdb_file}")
        return ""
    chain = model[chain_id]
    ppb = PPBuilder()
    peptides = ppb.build_peptides(chain)
    if not peptides:
        return ""
    seq = "".join([str(pp.get_sequence()) for pp in peptides])
    return seq


def calculate_ca_distance_matrix(ca_coords):
    """
    计算Cα原子之间的欧式距离矩阵。
    """
    num_atoms = len(ca_coords)
    if num_atoms == 0:
        return np.array([])
    # 使用高效的 numpy 广播计算
    dist_matrix = np.linalg.norm(ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :], axis=2)
    return dist_matrix


############################################
# Step 4: Sequence alignment and distance weighting
############################################

def align_sequences(seq1, seq2):
    """
    使用全局比对对齐两个序列，并返回最佳对齐结果。
    """
    alignments = pairwise2.align.globalxx(seq1, seq2)
    if not alignments:
        return None, None
    best_alignment = alignments[0]
    aligned_seq1 = best_alignment.seqA
    aligned_seq2 = best_alignment.seqB
    return aligned_seq1, aligned_seq2


def integrate_sequence_info(distance_matrix, aligned_seq1, aligned_seq2, similarity_score):
    """
    根据序列相似度对距离矩阵进行加权。
    在此示例中，我们简单地对每个匹配的残基对应用一个基于序列相似的加权。
    对齐中，相同的残基对（非gap）分配一个与overall相似度相关的加权因子。

    简单策略:
    - 如果aligned_seq1[i]和aligned_seq2[i]都是氨基酸(非'-')，则对第i个残基位置的距离值进行修正。
    - 使用 (1 - similarity_score) 将距离缩放，表示越相似的序列越强调原有距离（或相反，根据研究目的调整）。
    - 您可根据需要设计更复杂的加权函数，将相似度映射到更细粒度的残基权重上。

    注意：这里假设 PDB序列 与 distance_matrix 的对应为顺序一一对应（i.e. i-th CA对应序列的第i位残基）。
    如果PDB序列和对齐序列长度存在gap，需要严格映射残基索引，下面代码做了对应映射。
    """
    # 构建映射：aligned_seq1中的每个位置对应PDB序列的哪个残基
    original_idx = 0
    pdb_map = []  # 存储aligned_seq1的每个非gap位对应PDB序列中的index
    for char in aligned_seq1:
        if char != '-':
            pdb_map.append(original_idx)
            original_idx += 1
        else:
            pdb_map.append(None)

    # 对齐中共存的非gap对位置集合
    for i, (a1, a2) in enumerate(zip(aligned_seq1, aligned_seq2)):
        if a1 != '-' and a2 != '-':
            # 对应的PDB残基索引
            pdb_res_idx = pdb_map[i]
            if pdb_res_idx is not None:
                # 根据相似度加权距离矩阵中该残基与其他残基的距离
                # 例如: distance = distance * (1 - similarity_score)
                distance_matrix[pdb_res_idx, :] *= (1 - similarity_score)
                distance_matrix[:, pdb_res_idx] *= (1 - similarity_score)

    return distance_matrix


############################################
# Step 5: Main Processing Function
############################################

def process_pdb_folder(pdb_folder, output_folder, blast_result_file, fasta_dict, chain_id='A'):
    """
    处理PDB文件夹，计算所有PDB文件的距离矩阵并保存为npy格式。
    包含PDB加载、原子距离计算与进化信息优化。

    Args:
        pdb_folder (str): PDB文件所在的文件夹路径
        output_folder (str): 保存npy文件的输出文件夹路径
        blast_result_file (str): 计算进化权重矩阵的BLAST结果CSV文件路径
        fasta_dict (dict): {seq_id: sequence}
        chain_id (str): PDB文件中要提取的链ID
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
    blast_df = read_blast_results(blast_result_file)
    vdia_ids = set(fasta_dict.keys())

    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_folder, pdb_file)
        current_seq_id = os.path.splitext(pdb_file)[0]

        print(f"处理文件：{pdb_file}, 序列ID: {current_seq_id}")

        # 提取Cα坐标和PDB序列
        ca_coords, residue_ids = extract_ca_coordinates(pdb_path)
        if len(ca_coords) == 0:
            print(f"{pdb_file} 未找到CA原子，跳过。")
            continue

        pdb_seq = get_pdb_sequence(pdb_path, chain_id=chain_id)
        if len(pdb_seq) == 0:
            print(f"{pdb_file} 未能提取有效序列，跳过。")
            continue

        dist_matrix = calculate_ca_distance_matrix(ca_coords)
        if dist_matrix.size == 0:
            print(f"{pdb_file} 距离矩阵计算失败，跳过。")
            continue

        # 从BLAST结果中找到与current_seq_id匹配的序列
        match_id, best_e_value = get_best_match_sequence_id(current_seq_id, blast_df, vdia_ids)
        if match_id is None:
            print(f"{pdb_file} 在BLAST结果中未找到匹配序列ID，使用未加权距离矩阵。")
            # 直接保存
            np.save(os.path.join(output_folder, f"{current_seq_id}.npy"), dist_matrix)
            continue

        # 将E-value转换为相似度分值
        similarity_score = evalue_to_similarity(best_e_value)

        # 获取match_id对应的序列
        match_seq = fasta_dict.get(match_id, None)
        if match_seq is None:
            print(f"无法获取 {match_id} 的序列信息，使用未加权距离矩阵保存。")
            np.save(os.path.join(output_folder, f"{current_seq_id}.npy"), dist_matrix)
            continue

        # 序列对齐
        aligned_seq1, aligned_seq2 = align_sequences(pdb_seq, match_seq)
        if aligned_seq1 is None or aligned_seq2 is None:
            print(f"序列对齐失败，使用未加权距离矩阵保存。")
            np.save(os.path.join(output_folder, f"{current_seq_id}.npy"), dist_matrix)
            continue

        # 将序列相似信息融入距离矩阵
        final_dist_matrix = integrate_sequence_info(dist_matrix, aligned_seq1, aligned_seq2, similarity_score)

        # 保存最终矩阵
        out_file = os.path.join(output_folder, f"{current_seq_id}.npy")
        np.save(out_file, final_dist_matrix)
        print(f"{pdb_file} 已保存加权后的距离矩阵：{out_file}")


############################################
# Step 6: Implement get_sequence_by_id
############################################

def get_sequence_by_id(seq_id, fasta_dict):
    """
    获取给定seq_id的序列。此处从预加载的字典中获取。

    Args:
        seq_id (str): 序列ID
        fasta_dict (dict): {seq_id: sequence}
    Returns:
        str: 序列字符串
    """
    return fasta_dict.get(seq_id, None)


############################################
# Step 7: Usage Example
############################################

def main():
    # Step 1: Load V.dahliae sequences
    fasta_file = ""  # 请替换为您的FASTA文件路径
    fasta_dict = load_vdia_sequences(fasta_file)

    # Step 2: Process PDB folder with the BLAST CSV
    pdb_folder = ""        # 请替换为您的PDB文件夹路径
    output_folder = "" # 请替换为您希望输出的距离矩阵文件夹路径
    blast_result_file = ""  # 请替换为您的CSV文件路径

    process_pdb_folder(pdb_folder, output_folder, blast_result_file, fasta_dict, chain_id='A')


if __name__ == "__main__":
    main()
