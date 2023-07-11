import os
import numpy as np
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures


chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)

charge_list = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5, 0]
degree_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
valence_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]
hybridization_list = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "S", "UNSPECIFIED"]
hydrogen_list = [1, 2, 3, 4, 5, 6, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]

bond_list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def _DA(mol):
    D_list, A_list = [], []
    for feat in chem_feature_factory.GetFeaturesForMol(mol):
        if feat.GetFamily() == "Donor":
            D_list.append(feat.GetAtomIds()[0])
        if feat.GetFamily() == "Acceptor":
            A_list.append(feat.GetAtomIds()[0])

    return D_list, A_list


def _chirality(atom):
    if atom.HasProp("Chirality"):
        c_list = [
            (atom.GetProp("Chirality") == "Tet_CW"),
            (atom.GetProp("Chirality") == "Tet_CCW"),
        ]
    else:
        c_list = [0, 0]

    return c_list


def _stereochemistry(bond):
    if bond.HasProp("Stereochemistry"):
        s_list = [
            (bond.GetProp("Stereochemistry") == "Bond_Cis"),
            (bond.GetProp("Stereochemistry") == "Bond_Trans"),
        ]
    else:
        s_list = [0, 0]

    return s_list


def add_mol(mol_dict, mol):
    n_node = mol.GetNumAtoms()
    n_edge = mol.GetNumBonds() * 2

    D_list, A_list = _DA(mol)
    atom_fea1 = np.eye(118, dtype=bool)[[a.GetAtomicNum() for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(charge_list), dtype=bool)[
        [charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]
    ][:, :-1]
    atom_fea3 = np.eye(len(degree_list), dtype=bool)[
        [degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]
    ][:, :-1]
    atom_fea4 = np.eye(len(hybridization_list), dtype=bool)[
        [hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]
    ][:, :-2]
    atom_fea5 = np.eye(len(hydrogen_list), dtype=bool)[
        [
            hydrogen_list.index(a.GetTotalNumHs(includeNeighbors=True))
            for a in mol.GetAtoms()
        ]
    ][:, :-1]
    atom_fea6 = np.eye(len(valence_list), dtype=bool)[
        [valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]
    ][:, :-1]
    atom_fea7 = np.array(
        [[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())],
        dtype=bool,
    )
    atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype=bool)
    atom_fea9 = np.array(
        [[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()],
        dtype=bool,
    )
    atom_fea10 = np.array(
        [[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype=bool
    )

    # formal charge compress (-2 ~ 2)
    atom_fea2[:, 1:5] = np.max(atom_fea2[:, 1:5], axis=1).reshape(-1, 1)
    atom_fea2[:, 6:10] = np.max(atom_fea2[:, 6:10], axis=1).reshape(-1, 1)
    atom_fea2 = np.delete(atom_fea2, [2, 3, 4, 7, 8, 9], axis=1)

    # degree compress (1 ~ 6)
    atom_fea3[:, 5:] = np.max(atom_fea3[:, 5:], axis=1).reshape(-1, 1)
    atom_fea3 = np.delete(atom_fea3, [6, 7, 8, 9], axis=1)

    # no. hydrogen compress (1 ~ 4)
    atom_fea5[:, 3:] = np.max(atom_fea5[:, 3:], axis=1).reshape(-1, 1)
    atom_fea5 = np.delete(atom_fea5, [4, 5], axis=1)

    # valence compress (1 ~ 6)
    atom_fea6[:, 5:] = np.max(atom_fea6[:, 5:], axis=1).reshape(-1, 1)
    atom_fea6 = np.delete(atom_fea6, [6, 7, 8, 9, 10, 11], axis=1)

    node_attr = np.hstack(
        [
            atom_fea1,
            atom_fea2,
            atom_fea3,
            atom_fea4,
            atom_fea5,
            atom_fea6,
            atom_fea7,
            atom_fea8,
            atom_fea9,
            atom_fea10,
        ]
    )
    mol_dict["n_node"].append(n_node)
    mol_dict["n_edge"].append(n_edge)
    mol_dict["node_attr"].append(node_attr)

    if n_edge > 0:
        bond_fea1 = np.eye(len(bond_list), dtype=bool)[
            [bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]
        ]
        bond_fea2 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype=bool)
        bond_fea3 = np.array(
            [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()],
            dtype=bool,
        )

        edge_attr = np.hstack([bond_fea1, bond_fea2, bond_fea3])
        edge_attr = np.vstack([edge_attr, edge_attr])
        bond_loc = np.array(
            [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()],
            dtype=int,
        )
        src = np.hstack([bond_loc[:, 0], bond_loc[:, 1]])
        dst = np.hstack([bond_loc[:, 1], bond_loc[:, 0]])

        mol_dict["edge_attr"].append(edge_attr)
        mol_dict["src"].append(src)
        mol_dict["dst"].append(dst)

    return mol_dict


def add_dummy(mol_dict):
    n_node = 1
    n_edge = 0
    node_attr = np.zeros((1, 155))
    mol_dict["n_node"].append(n_node)
    mol_dict["n_edge"].append(n_edge)
    mol_dict["node_attr"].append(node_attr)

    return mol_dict


def dict_list_to_numpy(mol_dict):
    mol_dict["n_node"] = np.array(mol_dict["n_node"]).astype(int)
    mol_dict["n_edge"] = np.array(mol_dict["n_edge"]).astype(int)
    mol_dict["node_attr"] = np.vstack(mol_dict["node_attr"]).astype(bool)
    if np.sum(mol_dict["n_edge"]) > 0:
        mol_dict["edge_attr"] = np.vstack(mol_dict["edge_attr"]).astype(bool)
        mol_dict["src"] = np.hstack(mol_dict["src"]).astype(int)
        mol_dict["dst"] = np.hstack(mol_dict["dst"]).astype(int)
    else:
        mol_dict["edge_attr"] = np.empty((0, len(bond_list) + 4)).astype(bool)
        mol_dict["src"] = np.empty(0).astype(int)
        mol_dict["dst"] = np.empty(0).astype(int)

    return mol_dict
