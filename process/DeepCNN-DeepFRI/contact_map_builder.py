#contact_map_builder.py
import numpy as np
from Bio import Align
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.SeqUtils import seq1

TEN_ANGSTROMS     = 10.0
ALIGNED_BY_SEQRES = 'aligned by SEQRES'
ATOMS_ONLY        = 'ATOM lines only'
INCOMPARABLE_PAIR = 10000.
KEY_NOT_FOUND     = 1000.

class ContactMapContainer:
    def __init__(self):
        self.chains = {}

    def with_chain(self, chain_name):
        self.chains[chain_name] = {}

    def with_chain_seq(self, chain_name, seq):
        self.chains[chain_name]['seq'] = seq

    def with_map_for_chain(self, chain_name, contact_map):
        self.chains[chain_name]['contact-map'] = contact_map

    def with_alignment_for_chain(self, chain_name, alignment):
        self.chains[chain_name]['alignment'] = alignment

    def with_method_for_chain(self, chain_name, method):
        self.chains[chain_name]['method'] = method

    def with_final_seq_for_chain(self, chain_name, final_seq):
        self.chains[chain_name]['final-seq'] = final_seq


def correct_residue(x, target):
    try:
        sl = protein_letters_3to1[x.resname]
        if sl == target:
            return True
        return False
    except KeyError:
        return False


class DistanceMapBuilder:
    def __init__(self,
                 atom="CA",
                 verbose=True,
                 pedantic=True,
                 glycine_hack=-1):

        self.verbose = verbose
        self.pedantic = pedantic
        self.set_atom(atom)
        if not isinstance(glycine_hack, (int, float)):
            raise ValueError(f"{glycine_hack} is not an int")
        self.glycine_hack = glycine_hack

    def speak(self, *args, **kwargs):
        """
        Print a message or blackhole it
        """
        if self.verbose:
            print(*args, **kwargs)

    def set_atom(self, atom):
        if atom.casefold() not in ['ca', 'cb']:
            raise ValueError(f"{atom.casefold()} not 'ca' or 'cb'")
        self.__atom = atom.upper()
        return self

    @property
    def atom(self):
        return self.__atom

    def generate_map_for_pdb(self, structure_container):
        contact_maps = ContactMapContainer()
        model = structure_container.structure[0]

        for chain_name in structure_container.chains:
            chain = structure_container.chains[chain_name]
            contact_maps.with_chain(chain_name)
            self.speak(f"\nProcessing chain {chain_name}")

            seqres_seq = chain.get('seqres-seq', '')
            atom_seq = chain.get('atom-seq', '')

            # 直接使用ATOM序列长度，跳过对齐
            contact_maps.with_method_for_chain(chain_name, ATOMS_ONLY)
            residues = list(model[chain_name].get_residues())

            final_residue_list = []
            for residue in residues:
                try:
                    # 检查是否存在目标原子（如CA）
                    _ = residue[self.atom]
                    final_residue_list.append(residue)
                except KeyError:
                    # 跳过缺失目标原子的残基
                    continue

            # 动态计算距离矩阵大小
            n = len(final_residue_list)
            dist_matrix = np.full((n, n), fill_value=-1, dtype=np.float32)
            for i in range(n):
                for j in range(i, n):
                    dist = self.__calc_residue_dist(final_residue_list[i], final_residue_list[j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist

            contact_maps.with_map_for_chain(chain_name, dist_matrix)
            contact_maps.with_chain_seq(chain_name, atom_seq)

        return contact_maps

    def __residue_list_to_contact_map(self, residue_list, length):
        dist_matrix = self.__calc_dist_matrix(residue_list)
        diag = self.__diagnolize_to_fill_gaps(dist_matrix, length)
        #contact_map = self.__create_adj(diag, TEN_ANGSTROMS)
        contact_map = diag
        return contact_map

    def __norm_adj(self, A):
        #  Normalize adj matrix.
        with np.errstate(divide='ignore'):
            d = 1.0 / np.sqrt(A.sum(axis=1))
        d[np.isinf(d)] = 0.0

        # normalize adjacency matrices
        d = np.diag(d)
        A = d.dot(A.dot(d))

        return A

    def __create_adj(self, _A, thresh):
        # Create CMAP from distance
        A = _A.copy()
        with np.errstate(invalid='ignore'):
            A[A <= thresh] = 1.0
            A[A > thresh] = 0.0
            A[np.isnan(A)] = 0.0
            A = self.__norm_adj(A)

        return A

    def __calc_residue_dist(self, residue_one, residue_two):
        """Returns the `self.atom` distance between two residues"""
        if bool({residue_one, residue_two} & {None}):
            return INCOMPARABLE_PAIR
        try:
            dist = self.__euclidean(residue_one, self.atom,
                                    residue_two, self.atom)
        except KeyError:
            if self.atom == "CB":
                if self.glycine_hack < 0: # CA-mode for CB+GLY
                    try:
                        dist = self.__euclidean(residue_one,'CA',
                                                residue_two,'CA')
                    except KeyError:
                        dist = KEY_NOT_FOUND
                else:
                    dist = self.glycine_hack
            else:
                dist = KEY_NOT_FOUND
        return dist

    def __euclidean(self, res1, atom1, res2, atom2):
        diff = res1[atom1] - res2[atom2]
        return np.sqrt(np.sum(diff * diff))


    def __diagnolize_to_fill_gaps(self, distance_matrix, length):
        # Create CMAP from distance
        A = distance_matrix.copy()
        for i in range(length):
            if A[i][i] == INCOMPARABLE_PAIR:
                A[i][i] = 1.0
                try:
                    A[i + 1][i] = 1.0
                except IndexError:
                    pass
                try:
                    A[i][i + 1] = 1.0
                except IndexError:
                    pass

        return A

    def __calc_dist_matrix(self, chain_one):
        """Returns a matrix of C-alpha distances between two chains"""
        answer = np.zeros((len(chain_one), len(chain_one)), np.float)
        for row, residue_one in enumerate(chain_one):
            for col, residue_two in enumerate(chain_one[row:], start=row):
                if col >= len(chain_one):
                    continue  # enumerate syntax is convenient, but results in invalid indices on last column
                answer[row, col] = self.__calc_residue_dist(residue_one, residue_two)
                answer[col, row] = answer[row, col]  # cchandler
        return answer
