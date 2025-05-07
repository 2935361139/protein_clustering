#structure_file_reader.py
import json
import os
import tempfile
import re
from Bio.PDB import MMCIFParser, PDBParser
from Bio.Data.PDBData import protein_letters_3to1
import warnings
import Bio
from Bio import SeqIO
from Bio.Data.SCOPData import protein_letters_3to1


class PdbSeqResDataParser:
    def __init__(self, handle, parser_mode, chain_name, verbose=False):
        self.seq_res_seqs = []
        self.idx_to_chain = {}
        self.chain_count = 0
        # self.chain_ids = []
        for record in SeqIO.parse(handle, f"{parser_mode}-seqres"):
            if verbose:
                print("Record id %s, chain %s, len %s" % (record.id, record.annotations["chain"], len(record.seq)))
                print(record.dbxrefs)
                print(record.seq)
            if record.annotations['chain'] == chain_name:
                self.seq_res_seqs.append(record.seq)
                self.idx_to_chain[self.chain_count] = record.annotations['chain']
                # self.chain_ids.append(record.name)
                self.chain_count += 1
                break

    def has_seq_res_data(self):
        return self.chain_count > 0


class PdbAtomDataParser:
    def __init__(self, handle, parser_mode, chain_name, verbose=False):
        self.idx_to_chain = {}
        self.chain_to_idx = {}
        self.atom_seqs = []
        self.chain_count = 0

        for record in SeqIO.parse(handle, f"{parser_mode}-atom"):
            if verbose:
                print("Record id %s, chain %s len %s" % (record.id, record.annotations["chain"], len(record.seq)))
                print(record.seq)
            if record.annotations['chain'] == chain_name:
                self.atom_seqs.append(record.seq)
                self.idx_to_chain[self.chain_count] = record.annotations['chain']
                self.chain_to_idx[record.annotations['chain']] = self.chain_count
                self.chain_count += 1
                break


class StructureContainer:
    def __init__(self):
        self.structure = None
        self.chains = {}
        self.id_code = None

    def with_id_code(self, id_code):
        self.id_code = id_code
        return self

    def with_structure(self, structure):
        self.structure = structure
        return self

    def with_chain(self, chain_name, seqres_seq, atom_seq):
        chain_info = {'seqres-seq': seqres_seq, 'atom-seq': atom_seq}
        if seqres_seq is not None:
            chain_info['seq'] = seqres_seq
        else:
            chain_info['seq'] = atom_seq
        self.chains[chain_name] = chain_info
        return self

    def with_seqres(self,seqres_seq):
        for chain_name in self.chains:
            self.chains[chain_name]['seqres-seq'] = seqres_seq
        return self

    def toJSON(self):
        result = {'chain_info': self.chains, 'id_code': self.id_code}
        return json.dumps(result, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4, skipkeys=True)

def build_structure_container_for_pdb(structure_data, chain_name):
    is_cif = re.search('^_', structure_data, flags=re.MULTILINE) is not None
    parser_mode = 'cif' if is_cif else 'pdb'
    target_chain = str(chain_name).strip()

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
        temp.write(structure_data)
        temp_path = temp.name

    container_builder = StructureContainer()

    try:
        if parser_mode == 'cif':
            # 使用MMCIFParser解析结构
            parser = MMCIFParser()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 忽略Biopython警告
                structure = parser.get_structure("input", temp_path)
            mmcif_dict = parser._mmcif_dict

            # 提取SEQRES序列
            seqres_seq = None
            if "_entity_poly.pdbx_seq_one_letter_code" in mmcif_dict:
                seqres_seq = mmcif_dict["_entity_poly.pdbx_seq_one_letter_code"][0].replace("\n", "")

            # 提取ATOM序列（基于CA原子）
            atom_seq = []
            for model in structure:
                for chain in model:
                    if chain.id == target_chain:
                        for residue in chain:
                            if "CA" in residue:
                                resname = residue.resname
                                one_letter = protein_letters_3to1.get(resname, "X")
                                atom_seq.append(one_letter)
                        break
            atom_seq = "".join(atom_seq)
            id_code = os.path.basename(temp_path).split(".")[0].upper()
        else:
            # PDB解析逻辑（保持不变）
            parser = PDBParser()
            structure = parser.get_structure("input", temp_path)
            id_code = structure.header.get("idcode", "UNKNOWN")
            # ...（原有PDB解析代码）

        container_builder.with_id_code(id_code)
        container_builder.with_chain(target_chain, seqres_seq, atom_seq)
        container_builder.with_structure(structure)
    finally:
        os.remove(temp_path)

    return container_builder
