#PDB2distMap.py

from create_nrPDB_GO_annot import read_fasta, load_clusters
from structure_file_reader import build_structure_container_for_pdb
from contact_map_builder import DistanceMapBuilder
from Bio.PDB import PDBList
import requests
from functools import partial
import numpy as np
import argparse
import csv
import os
import gzip
import io
from Bio.PDB import PDBList
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def make_distance_maps(pdbfile, chain=None, sequence=None):
    pdb_handle = open(pdbfile, 'r')
    structure_container = build_structure_container_for_pdb(pdb_handle.read(), chain).with_seqres(sequence)
    pdb_handle.close()

    # 验证残基数量
    chain_info = structure_container.chains.get(chain, {})
    seqres_len = len(chain_info.get('seqres-seq', ''))
    atom_len = len(chain_info.get('atom-seq', ''))
    if seqres_len != atom_len:
        print(f"WARNING: SEQRES length ({seqres_len}) != ATOM length ({atom_len}) for {pdbfile} chain {chain}")

    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)
    ca = mapper.generate_map_for_pdb(structure_container)
    cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)

    return ca.chains, cb.chains


def load_GO_annot(filename):
    """ Load GO annotations """
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                prot2annot[prot][onts[i]] = [goterm for goterm in prot_goterms[i].split(',') if goterm != '']
    return prot2annot, goterms, gonames


def load_EC_annot(filename):
    """ Load EC annotations """
    prot2annot = {}
    ec_numbers = []
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = next(reader)
        next(reader, None)  # skip the headers
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            prot2annot[prot] = [ec_num for ec_num in prot_ec_numbers.split(',')]
    return prot2annot, ec_numbers


def retrieve_pdb(pdb, chain, chain_seqres, pdir):
    """
    从指定的RCSB镜像下载并解压.cif.gz文件
    """
    pdb = pdb.lower()
    subdir = pdb[1:3]  # 例如：2wst -> 'ws'
    cif_gz_url = f"https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{subdir}/{pdb}.cif.gz"
    local_gz_path = os.path.join(pdir, f"{pdb}.cif.gz")
    local_cif_path = os.path.join(pdir, f"{pdb}.cif")

    # 确保临时目录存在
    os.makedirs(pdir, exist_ok=True)

    try:
        # 下载并解压 .cif.gz 文件
        response = requests.get(cif_gz_url, stream=True, verify=False)
        if response.status_code == 200:
            # 直接解压到内存，避免写入.gz文件
            with gzip.open(io.BytesIO(response.content), 'rb') as gz_file:
                with open(local_cif_path, 'wb') as out_file:
                    out_file.write(gz_file.read())
            print(f"Downloaded and extracted: {cif_gz_url}")
        else:
            print(f"Failed to download {cif_gz_url}: HTTP {response.status_code}")
            return None, None

        try:
            # 生成原始接触图
            ca, cb = make_distance_maps(local_cif_path, chain=chain, sequence=chain_seqres)
            ca_map = ca[chain]['contact-map']
            cb_map = cb[chain]['contact-map']

            # === 新增：强制统一尺寸到160x160 ===
            FIXED_SIZE = 160

            def fix_matrix_size(matrix):
                """将矩阵裁剪/填充到固定尺寸"""
                current_size = matrix.shape[0]
                if current_size > FIXED_SIZE:
                    # 裁剪中心部分
                    start = (current_size - FIXED_SIZE) // 2
                    return matrix[start:start + FIXED_SIZE, start:start + FIXED_SIZE]
                else:
                    # 对称填充0
                    pad = FIXED_SIZE - current_size
                    return np.pad(matrix, ((0, pad), (0, pad)), mode='constant')

            fixed_ca = fix_matrix_size(ca_map)
            fixed_cb = fix_matrix_size(cb_map)
            # ===============================

            # === 处理序列长度 ===
            # 截断或填充序列到160
            processed_seq = chain_seqres[:FIXED_SIZE] if len(chain_seqres) >= FIXED_SIZE else chain_seqres.ljust(
                FIXED_SIZE, '-')

            return fixed_ca, fixed_cb, processed_seq  # 返回处理后的序列
        except Exception as e:
            print(f"Error processing {pdb}: {e}")
            return None, None, None
    finally:
        # 清理临时文件（可选）
        if os.path.exists(local_cif_path):
            os.remove(local_cif_path)  # 删除解压后的.cif文件（如果需要保留可注释此行）
        if os.path.exists(local_gz_path):
            os.remove(local_gz_path)

def load_list(fname):
    """
    Load PDB chains
    """
    pdb_chain_list = set()
    fRead = open(fname, 'r')
    for line in fRead:
        pdb_chain_list.add(line.strip())
    fRead.close()

    return pdb_chain_list


def write_annot_npz(prot, prot2seq=None, out_dir=None):
    pdb, chain = prot.split('-')
    print(f"Processing: {prot}")
    try:
        tmp_pdb_dir = os.path.join(out_dir, "tmp_PDB_files_dir")
        A_ca, A_cb, processed_seq = retrieve_pdb(pdb, chain, prot2seq[prot], pdir=tmp_pdb_dir)
        if A_ca is None or A_cb is None:
            print(f"Skipping {prot} due to download/processing error")
            return

        # 新增校验
        assert A_ca.shape == (160, 160), f"CA矩阵尺寸错误: {A_ca.shape}"
        assert A_cb.shape == (160, 160), f"CB矩阵尺寸错误: {A_cb.shape}"
        assert len(processed_seq) == 160, f"序列长度错误: {len(processed_seq)}"

        np.savez_compressed(
            os.path.join(out_dir, prot),
            C_alpha=A_ca,
            C_beta=A_cb,
            seqres=processed_seq,  # 使用处理后的序列
        )
    except Exception as e:
        print(f"Error processing {prot}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-annot', type=str, default='D:/bjfu/GNN-transformer/protein-predictor/data/yuanshuju/nrPDB-GO_2019.06.18_annot.tsv',help="Input file (*.tsv) with preprocessed annotations.")
    parser.add_argument('-ec', help="Use EC annotations.", action="store_true")
    parser.add_argument('-seqres', type=str, default='D:/bjfu/GNN-transformer/protein-predictor/output/pdb_chains.fasta', help="PDB chain seqres fasta.")
    parser.add_argument('-num_threads', type=int, default=4, help="Number of threads (CPUs) to use in the computation.")
    parser.add_argument('-bc', type=str, default='D:/bjfu/GNN-transformer/protein-predictor/data/-bc/bc-0.9.out', help="Clusters of PDB chains computd by Blastclust.")
    parser.add_argument('-out_dir', type=str, default='D:/bjfu/GNN-transformer/protein-predictor/data/output-npz/', help="Output directory with distance maps saved in *.npz format.")
    args = parser.parse_args()

    # load annotations
    prot2goterms = {}
    if args.annot is not None:
        if args.ec:
            prot2goterms, _ = load_EC_annot(args.annot)
        else:
            prot2goterms, _, _ = load_GO_annot(args.annot)
        print ("### number of annotated proteins: %d" % (len(prot2goterms)))

    # load sequences
    prot2seq = read_fasta(args.seqres)
    print ("### number of proteins with seqres sequences: %d" % (len(prot2seq)))

    # load clusters
    pdb2clust = {}
    if args.bc is not None:
        pdb2clust = load_clusters(args.bc)
        clusters = set([pdb2clust[prot][0] for prot in prot2goterms])
        print ("### number of annotated clusters: %d" % (len(clusters)))

    """
    # extracting unannotated proteins
    unannot_prots = set()
    for prot in pdb2clust:
        if (pdb2clust[prot][0] not in clusters) and (pdb2clust[prot][1] == 0) and (prot in prot2seq):
            unannot_prots.add(prot)
    print ("### number of unannot proteins: %d" % (len(unannot_prots)))
    """

    to_be_processed = set(prot2seq.keys())
    if len(prot2goterms) != 0:
        to_be_processed = to_be_processed.intersection(set(prot2goterms.keys()))
    if len(prot2goterms) != 0:
        to_be_processed = to_be_processed.intersection(set(pdb2clust.keys()))
    print ("Number of pdbs to be processed=", len(to_be_processed))
    print (to_be_processed)

    # 加载数据后添加：
    common_seq_annot = set(prot2seq.keys()) & set(prot2goterms.keys())
    print("prot2seq 和 prot2goterms 共有ID数量:", len(common_seq_annot))
    if args.bc:
        common_all = common_seq_annot & set(pdb2clust.keys())
        print("prot2seq、prot2goterms、pdb2clust 共有ID数量:", len(common_all))

    # process on multiple cpus
    nprocs = args.num_threads
    out_dir = args.out_dir
    import multiprocessing
    nprocs = np.minimum(nprocs, multiprocessing.cpu_count())
    if nprocs > 4:
        pool = multiprocessing.Pool(processes=nprocs)
        pool.map(partial(write_annot_npz, prot2seq=prot2seq, out_dir=out_dir),
                 to_be_processed)
    else:
        for prot in to_be_processed:
            write_annot_npz(prot, prot2seq=prot2seq, out_dir=out_dir)