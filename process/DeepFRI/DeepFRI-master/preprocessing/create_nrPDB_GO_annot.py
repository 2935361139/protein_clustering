from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import networkx as nx
import numpy as np
import argparse
import obonet
import gzip
import csv

# 脚本专注于创建一个非冗余（Non-redundant, NR）蛋白数据库（PDB）注释集，针对基因本体（Gene Ontology, GO）分类。

exp_evidence_codes = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'CURATED'])
root_terms = set(['GO:0008150', 'GO:0003674', 'GO:0005575'])


def read_fasta(fn_fasta):
    aa = set(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    prot2seq = {}
    if fn_fasta.endswith('gz'):
        handle = gzip.open(fn_fasta, "rt")
    else:
        handle = open(fn_fasta, "rt")

    for record in SeqIO.parse(handle, "fasta"):
        seq = str(record.seq)
        prot = record.id
        pdb, chain = prot.split('_') if '_' in prot else prot.split('-')
        prot = pdb.upper() + '-' + chain
        if len(seq) >= 60 and len(seq) <= 1000:
            if len((set(seq).difference(aa))) == 0:
                prot2seq[prot] = seq

    handle.close()
    return prot2seq


def write_prot_list(protein_list, filename):
    fWrite = open(filename, 'w')
    for p in protein_list:
        fWrite.write("%s\n" % (p))
    fWrite.close()


def write_fasta(fn, sequences):
    with open(fn, "w") as output_handle:
        for sequence in sequences:
            SeqIO.write(sequence, output_handle, "fasta")


def load_go_graph(fname):
    return obonet.read_obo(open(fname, 'r'))


def load_pdbs(sifts_fname):
    pdb_chains = set()
    with gzip.open(sifts_fname, mode='rt') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb = row[0].strip().upper()
            chain = row[1].strip()
            pdb_chains.add(pdb + '-' + chain)
    return pdb_chains


def load_clusters(fname):
    pdb2clust = {}
    c_ind = 1
    fRead = open(fname, 'r')
    for line in fRead:
        clust = line.strip().split()
        clust = [p.replace('_', '-') for p in clust]
        for rank, p in enumerate(clust):
            pdb2clust[p] = (c_ind, rank)
        c_ind += 1
    fRead.close()
    return pdb2clust


def nr_set(chains, pdb2clust):
    clust2chain = {}
    for chain in chains:
        if chain in pdb2clust:
            c_idx = pdb2clust[chain][0]
            if c_idx not in clust2chain:
                clust2chain[c_idx] = chain
            else:
                _chain = clust2chain[c_idx]
                if pdb2clust[chain][1] < pdb2clust[_chain][1]:
                    clust2chain[c_idx] = chain
    return set(clust2chain.values())


def read_sifts(fname, chains, go_graph):
    print ("### Loading SIFTS annotations...")
    pdb2go = {}
    go2info = {}
    with gzip.open(fname, mode='rt') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb = row[0].strip().upper()
            chain = row[1].strip()
            evidence = row[4].strip()
            go_id = row[5].strip()
            pdb_chain = pdb + '-' + chain
            if (pdb_chain in chains) and (go_id in go_graph) and (go_id not in root_terms):
                if pdb_chain not in pdb2go:
                    pdb2go[pdb_chain] = {'goterms': [go_id], 'evidence': [evidence]}
                namespace = go_graph.nodes[go_id]['namespace']
                go_ids = nx.descendants(go_graph, go_id)
                go_ids.add(go_id)
                go_ids = go_ids.difference(root_terms)
                for go in go_ids:
                    pdb2go[pdb_chain]['goterms'].append(go)
                    pdb2go[pdb_chain]['evidence'].append(evidence)
                    name = go_graph.nodes[go]['name']
                    if go not in go2info:
                        go2info[go] = {'ont': namespace, 'goname': name, 'pdb_chains': set([pdb_chain])}
                    else:
                        go2info[go]['pdb_chains'].add(pdb_chain)
    return pdb2go, go2info


def write_output_files(fname, pdb2go, go2info, pdb2seq):
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    selected_goterms = {ont: set() for ont in onts}
    selected_proteins = set()
    for goterm in go2info:
        prots = go2info[goterm]['pdb_chains']
        num = len(prots)
        namespace = go2info[goterm]['ont']
        if num > 49 and num <= 5000:
            selected_goterms[namespace].add(goterm)
            selected_proteins = selected_proteins.union(prots)

    selected_goterms_list = {ont: list(selected_goterms[ont]) for ont in onts}
    selected_gonames_list = {ont: [go2info[goterm]['goname'] for goterm in selected_goterms_list[ont]] for ont in onts}

    for ont in onts:
        print ("###", ont, ":", len(selected_goterms_list[ont]))

    sequences_list = []
    protein_list = []
    with open(fname + '_annot.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for ont in onts:
            tsv_writer.writerow(["### GO-terms (%s)" % (ont)])
            tsv_writer.writerow(selected_goterms_list[ont])
            tsv_writer.writerow(["### GO-names (%s)" % (ont)])
            tsv_writer.writerow(selected_gonames_list[ont])
        tsv_writer.writerow(["### PDB-chain", "GO-terms (molecular_function)", "GO-terms (biological_process)", "GO-terms (cellular_component)"])
        for chain in selected_proteins:
            goterms = set(pdb2go[chain]['goterms'])
            if len(goterms) > 2:
                mf_goterms = goterms.intersection(set(selected_goterms_list[onts[0]]))
                bp_goterms = goterms.intersection(set(selected_goterms_list[onts[1]]))
                cc_goterms = goterms.intersection(set(selected_goterms_list[onts[2]]))
                if len(mf_goterms) > 0 or len(bp_goterms) > 0 or len(cc_goterms) > 0:
                    sequences_list.append(SeqRecord(Seq(pdb2seq[chain]), id=chain, description="nrPDB"))
                    protein_list.append(chain)
                    tsv_writer.writerow([chain, ','.join(mf_goterms), ','.join(bp_goterms), ','.join(cc_goterms)])

    np.random.seed(1234)
    np.random.shuffle(protein_list)
    print ("Total number of annot nrPDB=%d" % (len(protein_list)))

    test_list = set()
    test_sequences_list = []
    i = 0
    while len(test_list) < 5000 and i < len(protein_list):
        goterms = pdb2go[protein_list[i]]['goterms']
        evidence = pdb2go[protein_list[i]]['evidence']
        goterm2evidence = {goterms[i]: evidence[i] for i in range(len(goterms))}

        mf_goterms = set(goterms).intersection(set(selected_goterms_list[onts[0]]))
        bp_goterms = set(goterms).intersection(set(selected_goterms_list[onts[1]]))
        cc_goterms = set(goterms).intersection(set(selected_goterms_list[onts[2]]))

        mf_evidence = [goterm2evidence[goterm] for goterm in mf_goterms]
        mf_evidence = [1 if evid in exp_evidence_codes else 0 for evid in mf_evidence]

        bp_evidence = [goterm2evidence[goterm] for goterm in bp_goterms]
        bp_evidence = [1 if evid in exp_evidence_codes else 0 for evid in bp_evidence]

        cc_evidence = [goterm2evidence[goterm] for goterm in cc_goterms]
        cc_evidence = [1 if evid in exp_evidence_codes else 0 for evid in cc_evidence]

        if len(mf_goterms) > 0 and len(bp_goterms) > 0 and len(cc_goterms) > 0:
            if sum(mf_evidence) > 0 and sum(bp_evidence) > 0 and sum(cc_evidence) > 0:
                test_list.add(protein_list[i])
                test_sequences_list.append(SeqRecord(Seq(pdb2seq[protein_list[i]]), id=protein_list[i], description="nrPDB_test"))
        i += 1

    print ("Total number of test nrPDB=%d" % (len(test_list)))

    protein_list = list(set(protein_list).difference(test_list))
    np.random.shuffle(protein_list)

    idx = int(0.9*len(protein_list))
    write_prot_list(test_list, fname + '_test.txt')
    write_prot_list(protein_list[:idx], fname + '_train.txt')
    write_prot_list(protein_list[idx:], fname + '_valid.txt')
    write_fasta(fname + '_sequences.fasta', sequences_list)
    write_fasta(fname + '_test_sequences.fasta', test_sequences_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sifts', type=str, default='./data/pdb_chain_go_2019.06.18.tsv.gz', help="SIFTS annotation files.")
    # 这是一个通常以.tsv压缩格式提供的文件，包含PDB条目与基因本体(GO)
    # 注释的映射。这个文件可以从EBI的SIFTS项目下载。
    parser.add_argument('-bc', type=str, default='./data/bc-95.out', help="Blastclust of PDB chains.")
    # 指的是Blastclust生成的聚类文件，这个文件包含基于序列相似度聚类的PDB链。这有助于建立非冗余的蛋白数据库。
    # 如果没有可用的聚类数据，这个参数可以忽略，或者需要你自己先使用序列聚类工具（如CD - HIT或Blastclust）处理PDB序列。
    parser.add_argument('-seqres', type=str, default='./data/pdb_seqres.txt.gz', help="PDB chain seqres fasta.")
    #指向包含PDB条目序列的fasta文件，通常名为pdb_seqres.txt。这个文件包含了PDB数据库中所有结构的序列信息，可从RCSBPDB官网下载。
    parser.add_argument('-obo', type=str, default='./data/go-basic.obo', help="Gene Ontology hierarchy.")
    # GeneOntology(GO)文件，通常是以.obo格式提供。这是一个本体文件，描述了GO术语的层次结构和关系。可以从GeneOntology官方网站下载
    parser.add_argument('-out', type=str, default='./data/nrPDB-GO_2019.06.18', help="Output filename prefix.")
    args = parser.parse_args()

    annoted_chains = load_pdbs(args.sifts)
    pdb2clust = load_clusters(args.bc)
    pdb2seq = read_fasta(args.seqres)
    nr_chains = nr_set(annoted_chains, pdb2clust)
    print ("### nrPDB annotated chains=", len(nr_chains))

    go_graph = load_go_graph(args.obo)
    pdb2go, go2info = read_sifts(args.sifts, nr_chains, go_graph)

    write_output_files(args.out, pdb2go, go2info, pdb2seq)

# 主要功能和步骤
# 1数据加载与解析:
# 使用 read_fasta 从GZip压缩的FASTA文件中读取蛋白序列。
# load_pdbs 从SIFTS文件加载PDB链信息。
# load_clusters 从Blastclust输出文件中读取蛋白聚类信息，有助于识别和去除序列的冗余。
# 2基因本体（GO）图的加载:
# load_go_graph 使用 obonet 库读取OBO格式的GO文件，建立一个基因本体的网络图（networkx graph），用于后续的注释分析和继承性的确定。
# 3非冗余数据集的创建:
# nr_set 函数通过比较聚类信息和序列数据，创建一个非冗余的蛋白集合。
# 4GO注释提取:
# read_sifts 从SIFTS文件提取与PDB结构相关的GO注释，确保注释的精确性。
# 5输出文件生成:
# write_output_files 生成训练集、验证集和测试集等多个输出文件，记录EC号的覆盖率和数据集中蛋白的数量。这些文件可以直接用于机器学习模型的训练和测试。

# 在 create_nrPDB_GO_annot.py 脚本中，输出文件的具体内容和格式取决于脚本的具体实现。基于你之前提供的脚本内容，它可能会生成以下几类文件：
# 注释的.tsv文件 (_annot.tsv)：这是一个表格格式的文件，使用制表符分隔，可能包含如下信息：
# GO术语（每种本体类型如分子功能、生物过程、细胞组成各自列出）。
# 对应的PDB链ID和它们关联的GO术语。
# 每个GO术语的名称。

# 这个文件提供了关于哪些蛋白质链与哪些GO术语相关联的详细数据。
# 序列文件 (_sequences.fasta 和 _test_sequences.fasta)：这些是FASTA格式的文件，包含了用于后续分析或模型训练的蛋白质序列。
# 通常，_sequences.fasta 包含训练集的序列，而 _test_sequences.fasta 包含测试集的序列。
# 蛋白列表文件 (_test.txt, _train.txt, 和 _valid.txt)：这些文本文件包含了分割后用于训练、验证和测试的蛋白质链ID列表