import numpy as np
from rdkit import Chem

from src.preprocess_util import add_mol, add_dummy, dict_list_to_numpy


def mol_dict():
    return {
        "n_node": [],
        "n_edge": [],
        "node_attr": [],
        "edge_attr": [],
        "src": [],
        "dst": [],
    }


def get_graph_data(rsmi_list, yld_list, filename):
    rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_list])
    pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_list])

    rmol_dict = [mol_dict() for _ in range(rmol_max_cnt)]
    pmol_dict = [mol_dict() for _ in range(pmol_max_cnt)]

    reaction_dict = {"yld": [], "rsmi": []}

    print("--- generating graph data for %s" % filename)
    print(
        "--- n_reactions: %d, reactant_max_cnt: %d, product_max_cnt: %d"
        % (len(rsmi_list), rmol_max_cnt, pmol_max_cnt)
    )

    for i in range(len(rsmi_list)):
        rsmi = rsmi_list[i].replace("~", "-")
        yld = yld_list[i]

        [reactants_smi, products_smi] = rsmi.split(">>")

        # processing reactants
        reactants_smi_list = reactants_smi.split(".")
        for _ in range(rmol_max_cnt - len(reactants_smi_list)):
            reactants_smi_list.append("")
        for j, smi in enumerate(reactants_smi_list):
            if smi == "":
                rmol_dict[j] = add_dummy(rmol_dict[j])
            else:
                rmol = Chem.MolFromSmiles(smi)
                rs = Chem.FindPotentialStereo(rmol)
                for element in rs:
                    if (
                        str(element.type) == "Atom_Tetrahedral"
                        and str(element.specified) == "Specified"
                    ):
                        rmol.GetAtomWithIdx(element.centeredOn).SetProp(
                            "Chirality", str(element.descriptor)
                        )
                    elif (
                        str(element.type) == "Bond_Double"
                        and str(element.specified) == "Specified"
                    ):
                        rmol.GetBondWithIdx(element.centeredOn).SetProp(
                            "Stereochemistry", str(element.descriptor)
                        )

                rmol = Chem.RemoveHs(rmol)
                rmol_dict[j] = add_mol(rmol_dict[j], rmol)

        # processing products
        products_smi_list = products_smi.split(".")
        for _ in range(pmol_max_cnt - len(products_smi_list)):
            products_smi_list.append("")
        for j, smi in enumerate(products_smi_list):
            if smi == "":
                pmol_dict[j] = add_dummy(pmol_dict[j])
            else:
                pmol = Chem.MolFromSmiles(smi)
                ps = Chem.FindPotentialStereo(pmol)
                for element in ps:
                    if (
                        str(element.type) == "Atom_Tetrahedral"
                        and str(element.specified) == "Specified"
                    ):
                        pmol.GetAtomWithIdx(element.centeredOn).SetProp(
                            "Chirality", str(element.descriptor)
                        )
                    elif (
                        str(element.type) == "Bond_Double"
                        and str(element.specified) == "Specified"
                    ):
                        pmol.GetBondWithIdx(element.centeredOn).SetProp(
                            "Stereochemistry", str(element.descriptor)
                        )

                pmol = Chem.RemoveHs(pmol)
                pmol_dict[j] = add_mol(pmol_dict[j], pmol)

        # yield and reaction SMILES
        reaction_dict["yld"].append(yld)
        reaction_dict["rsmi"].append(rsmi)

        # monitoring
        if (i + 1) % 1000 == 0:
            print("--- %d/%d processed" % (i + 1, len(rsmi_list)))

    # datatype to numpy
    for j in range(rmol_max_cnt):
        rmol_dict[j] = dict_list_to_numpy(rmol_dict[j])
    for j in range(pmol_max_cnt):
        pmol_dict[j] = dict_list_to_numpy(pmol_dict[j])
    reaction_dict["yld"] = np.array(reaction_dict["yld"])

    # save file
    np.savez_compressed(filename, data=[rmol_dict, pmol_dict, reaction_dict])
