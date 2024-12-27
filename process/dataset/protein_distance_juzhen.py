import numpy as np
import os
from Bio.PDB import PDBParser, PPBuilder


def process_pdb_file(pdb_path):
    # 读取一个PDB文件，提取所有的Cα原子，并计算它们之间的距离矩阵
    parser = PDBParser()
    structure = parser.get_structure('PDB', pdb_path)
    ca_atoms = [atom for model in structure for chain in model for residue in chain if "CA" in residue for atom in
                residue if atom.name == "CA"]
    num_atoms = len(ca_atoms)
    distance_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(ca_atoms[i].coord - ca_atoms[j].coord)
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix


def process_pdb_directory(directory_path):
    # 遍历指定目录下的所有PDB文件，并对每个文件调用上述处理函数
    distance_matrices = {}
    pdb_files = [f for f in os.listdir(directory_path) if f.endswith('.pdb')]

    for pdb_file in pdb_files:
        pdb_path = os.path.join(directory_path, pdb_file)
        distance_matrix = process_pdb_file(pdb_path)
        distance_matrices[pdb_file] = distance_matrix

    return distance_matrices


def save_distance_matrices(distance_matrices, save_directory):
    # 保存距离矩阵
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for pdb_file, matrix in distance_matrices.items():
        save_path = os.path.join(save_directory, f"{pdb_file[:-4]}.npy")
        np.save(save_path, matrix)


# 你的PDB文件所在文件夹
directory_path = ''
# 保存距离矩阵的目录
save_directory = ''

distance_matrices = process_pdb_directory(directory_path)
save_distance_matrices(distance_matrices, save_directory)
