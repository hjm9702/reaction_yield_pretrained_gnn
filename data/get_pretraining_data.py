import pickle
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors

from src.preprocess_util import add_mol
import warnings

warnings.filterwarnings("ignore")

def preprocess(molsuppl, graph_save_path):
    length = len(molsuppl)

    mol_dict = {
        "n_node": [],
        "n_edge": [],
        "node_attr": [],
        "edge_attr": [],
        "src": [],
        "dst": [],
    }

    for i, mol in enumerate(molsuppl):
        mol = molsuppl[i]
        try:
            Chem.SanitizeMol(mol)
            si = Chem.FindPotentialStereo(mol)
            for element in si:
                if (
                    str(element.type) == "Atom_Tetrahedral"
                    and str(element.specified) == "Specified"
                ):
                    mol.GetAtomWithIdx(element.centeredOn).SetProp(
                        "Chirality", str(element.descriptor)
                    )
                elif (
                    str(element.type) == "Bond_Double"
                    and str(element.specified) == "Specified"
                ):
                    mol.GetBondWithIdx(element.centeredOn).SetProp(
                        "Stereochemistry", str(element.descriptor)
                    )
            assert "." not in Chem.MolToSmiles(mol)
        except:
            continue

        mol = Chem.RemoveHs(mol)
        mol_dict = add_mol(mol_dict, mol)

        if (i + 1) % 10000 == 0:
            print(f"{i+1}/{length} processed")

    mol_dict["n_node"] = np.array(mol_dict["n_node"]).astype(int)
    mol_dict["n_edge"] = np.array(mol_dict["n_edge"]).astype(int)
    mol_dict["node_attr"] = np.vstack(mol_dict["node_attr"]).astype(bool)
    mol_dict["edge_attr"] = np.vstack(mol_dict["edge_attr"]).astype(bool)
    mol_dict["src"] = np.hstack(mol_dict["src"]).astype(int)
    mol_dict["dst"] = np.hstack(mol_dict["dst"]).astype(int)

    for key in mol_dict.keys():
        print(key, mol_dict[key].shape, mol_dict[key].dtype)

    with open(graph_save_path + "pubchem_graph.npz", "wb") as f:
        pickle.dump([mol_dict], f, protocol=5)


def get_mordred(molsuppl, mordred_save_path):
    # mordred calculation can be further accelerated if setting multiprocessing environments
    calc = Calculator(descriptors, ignore_3D=True)

    mordred_list = []

    for i, mol in enumerate(molsuppl):
        mol = molsuppl[i]

        try:
            Chem.SanitizeMol(mol)
            si = Chem.FindPotentialStereo(mol)
            for element in si:
                if (
                    str(element.type) == "Atom_Tetrahedral"
                    and str(element.specified) == "Specified"
                ):
                    mol.GetAtomWithIdx(element.centeredOn).SetProp(
                        "Chirality", str(element.descriptor)
                    )
                elif (
                    str(element.type) == "Bond_Double"
                    and str(element.specified) == "Specified"
                ):
                    mol.GetBondWithIdx(element.centeredOn).SetProp(
                        "Stereochemistry", str(element.descriptor)
                    )
            assert "." not in Chem.MolToSmiles(mol)
        except:
            continue

        mol_attr = calc(mol).fill_missing(np.nan)
        mordred_list.append(mol_attr)

        if i % 10000 == 0:
            print("%d mol- mordred calculated!" % i)

    mordred_list = np.vstack(mordred_list)

    with open(
        mordred_save_path + "pubchem_mordred.npz",
        "wb",
    ) as f:
        pickle.dump([mordred_list], f, protocol=5)
