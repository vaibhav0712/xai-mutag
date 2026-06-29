import os
from rdflib import URIRef, RDF, Namespace, Graph
from torch_geometric.data import Data
from dataclasses import dataclass
import torch.nn.functional as F
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


ROOT_DIR = Path.cwd()
dataset_path = ROOT_DIR / "mutag-hetero"

if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset path {dataset_path} does not exist. Please ensure the dataset is downloaded and placed in the correct directory.")


#CONSTANT DEFINITIONS
CARC = Namespace("http://dl-learner.org/carcinogenesis#")


@dataclass
class Molecule:
    """
    Clean representation of one molecule.
    """
    id: str
    label: int
    atoms: list
    bonds: list

#HELPER FUNCTIONS
def short_name(uri):
    return str(uri).split("#")[-1]


# Load datasets
print("[INFO] Loading RDF graph, train and test sets...")
g  = Graph()
g.parse("mutag-hetero/mutag_stripped.nt", format="nt")
train_df = pd.read_csv("mutag-hetero/trainingSet.tsv", sep="\t")
test_df = pd.read_csv("mutag-hetero/testSet.tsv", sep="\t")


print("[INFO] Loaded RDF graph with {} triples.".format(len(g)))

# labels dictionary to store the labels for each molecule i.e {"bond_uri": mutagenic_label}
labels = {}
for _, row in train_df.iterrows():
    labels[URIRef(row["bond"])] = int(row["label_mutagenic"])

for _, row in test_df.iterrows():
    if URIRef(row["bond"]) not in labels:
        labels[URIRef(row["bond"])] = int(row["label_mutagenic"])
    else:
        assert labels[URIRef(row["bond"])] == int(row["label_mutagenic"]), f"Conflicting labels for {row['bond']}"
        
print("[INFO] Total labelled molecules:", len(labels))


def extract_molecule(graph, molecule_uri, label):

    molecule = Molecule(
        id=short_name(molecule_uri),
        label=label,
        atoms=[],
        bonds=[]
    )

    # RDF atom URI -> integer node index
    atom_to_idx = {}

    # Extract atoms
    atom_uris = sorted(graph.objects(molecule_uri, CARC.hasAtom), key=str)
    for atom_uri in atom_uris:

        atom_type = next(graph.objects(atom_uri, RDF.type), None)

        if atom_type is None:
            continue

        idx = len(molecule.atoms)

        atom_to_idx[atom_uri] = idx

        molecule.atoms.append(
            {
                "uri": short_name(atom_uri),
                "type": short_name(atom_type)
            }
        )

    # Extract bonds
    bond_uris = sorted(graph.objects(molecule_uri, CARC.hasBond), key=str)
    for bond_uri in bond_uris:

        bond_type = next(graph.objects(bond_uri, RDF.type), None)

        if bond_type is None:
            continue

        endpoints = sorted(graph.objects(bond_uri, CARC.inBond), key=str)

        if len(endpoints) != 2:
            print(f"Skipping malformed bond {bond_uri}")
            continue

        atom1, atom2 = endpoints

        if atom1 not in atom_to_idx or atom2 not in atom_to_idx:
            print(f"Bond references missing atom in {bond_uri}")
            continue

        molecule.bonds.append(
            (
                atom_to_idx[atom1],
                atom_to_idx[atom2],
                short_name(bond_type)
            )
        )

    return molecule


dataset = []

for molecule_uri in sorted(labels.keys(), key=str):

    dataset.append(
        extract_molecule(
            g,
            molecule_uri,
            labels[molecule_uri]
        )
    )

print("[INFO] Total molecules extracted:", len(dataset))

num_atoms = [len(m.atoms) for m in dataset]
num_bonds = [len(m.bonds) for m in dataset]

print(f"[INFO] Average atoms : {sum(num_atoms)/len(num_atoms):.2f}")
print(f"[INFO] Average bonds : {sum(num_bonds)/len(num_bonds):.2f}")


# I don't think this functions are used
atom_counter = Counter()
for mol in dataset:
    for atom in mol.atoms:
        atom_counter[atom["type"]] += 1
print(atom_counter)
bond_counter = Counter()
for mol in dataset:
    for _, _, bond in mol.bonds:
        bond_counter[bond] += 1

print(bond_counter)


print(f"[INFO] Encoding atoms and bonds...")
# Atom encoder
unique_atom_types = sorted({
    atom["type"]
    for mol in dataset
    for atom in mol.atoms
})

atom_encoder = {
    atom: idx
    for idx, atom in enumerate(unique_atom_types)
}

idx_to_atom = {
    idx: atom
    for atom, idx in atom_encoder.items()
}
unique_bond_types = sorted({
    bond
    for mol in dataset
    for _, _, bond in mol.bonds
})

bond_encoder = {
    bond: idx
    for idx, bond in enumerate(unique_bond_types)
}

idx_to_bond = {
    idx: bond
    for bond, idx in bond_encoder.items()
}

print(f"[INFO] Encoding complete. Unique atom types: {len(unique_atom_types)}, Unique bond types: {len(unique_bond_types)}")



def molecule_to_pyg(
    molecule,
    atom_encoder,
    bond_encoder
):
    # Node Features (x)
    atom_ids = [
        atom_encoder[atom["type"]]
        for atom in molecule.atoms
    ]

    atom_ids = torch.tensor(atom_ids, dtype=torch.long)

    x = F.one_hot(
        atom_ids,
        num_classes=len(atom_encoder)
    ).float()

    # Edge Index
    edge_index = []

    edge_type = []

    for src, dst, bond in molecule.bonds:

        # Undirected graph
        edge_index.append([src, dst])
        edge_index.append([dst, src])

        bond_id = bond_encoder[bond]

        edge_type.append(bond_id)
        edge_type.append(bond_id)

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    edge_type = torch.tensor(
        edge_type,
        dtype=torch.long
    )

    # Graph Label
    y = torch.tensor(
        [molecule.label],
        dtype=torch.long
    )

    # PyG Data
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y
    )

    # Extra metadata
    data.molecule_id = molecule.id
    data.atom_info = molecule.atoms
    data.bond_info = molecule.bonds

    return data

print('[INFO] Converting molecules to PyG Data objects...')
pyg_dataset = [
    molecule_to_pyg(
        mol,
        atom_encoder,
        bond_encoder
    )
    for mol in dataset
]

print('[INFO] Total graphs in PyG dataset:', len(pyg_dataset))

torch.save(pyg_dataset, dataset_path / "pyg_dataset.pt")
print(f"[INFO] PyG dataset saved to {dataset_path / 'pyg_dataset.pt'}") 