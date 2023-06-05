"""
In the context of Hilammps, a system is a collection of atoms and molecules.
A molecule itself is a collection of atoms which are connected by bonds.
A complete description of a molecule should be given by ("DescriptorType"s):
    - "atom"s
    - "bond"s
    - "angle"s
    - "dihedral"s
    - "improper"s
"bond"s, "angle"s, "dihedral"s, and "improper"s are called "TopologicalType"s.

TODO: float values for bond, angle, dihedral, improper
A description is considered complete if, given the above descriptors,
positions of all atoms in the molecule can be determined up to arbitrary
translations and rotations.
TODO: Solving positions from descriptors.
TODO: One-to-one mapping of topological descriptors to positions vs degenaracy.

All "atom"s, "bond"s, etc. are labeled by a integer or string ("LabelType").
In addition to label, "TopologicalType"s contain indices of the atoms in the
molecule that make that descriptor. The type system is as follows:
    - LabelType = int | str
    - AtomType = LabelType
    - BondType = (LabelType, int, int)
    - AngleType = (LabelType, int, int, int)
    - DihedralType = (LabelType, int, int, int, int)
    - ImproperType = (LabelType, int, int, int, int)

A "MoleculeType" is an abstract base class for all molecules.
It provides methods to get the descriptors of the molecule.
    - get_positions() -> np.ndarray
    - get_atoms() -> tuple[AtomType, ...]
    - get_bonds() -> tuple[BondType, ...]
    - get_angles() -> tuple[AngleType, ...]
    - get_dihedrals() -> tuple[DihedralType, ...]
    - get_impropers() -> tuple[ImproperType, ...]
An empty tuple is returned if the molecule does not contain the descriptor.

"""
from __future__ import annotations

import abc
import typing

import numpy as np

# Descriptor types
DescriptorType = typing.Literal[
    "atom", "bond", "angle", "dihedral", "improper"
]
descriptor_types: tuple[DescriptorType, ...] = typing.get_args(DescriptorType)
# Topological types
TopologicalType = typing.Literal["bond", "angle", "dihedral", "improper"]
topological_types: tuple[TopologicalType, ...] = typing.get_args(
    TopologicalType
)
# Type system
LabelType = int | str
AtomType = LabelType
BondType = tuple[LabelType, int, int]
AngleType = tuple[LabelType, int, int, int]
DihedralType = tuple[LabelType, int, int, int, int]
ImproperType = tuple[LabelType, int, int, int, int]


class MoleculeType(abc.ABC):
    # --- Interface methods ---
    @abc.abstractmethod
    def get_positions(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_atoms(self) -> tuple[AtomType, ...]:
        ...

    @abc.abstractmethod
    def get_bonds(self) -> tuple[BondType, ...]:
        ...

    @abc.abstractmethod
    def get_angles(self) -> tuple[AngleType, ...]:
        ...

    @abc.abstractmethod
    def get_dihedrals(self) -> tuple[DihedralType, ...]:
        ...

    @abc.abstractmethod
    def get_impropers(self) -> tuple[ImproperType, ...]:
        ...

    # --- Concrete methods ---

    def _get_labels(self, descriptor: DescriptorType) -> tuple[LabelType, ...]:
        raise NotImplementedError

    def _get_extra_per_atom(self, topo: TopologicalType) -> int:
        raise NotImplementedError

    def _get_extra_special_per_atom(self) -> int:
        # TODO: extra special per atom should be inferred from the molecule
        raise NotImplementedError


def _unique_tuple(
    labels: typing.Sequence[typing.Any],
) -> tuple[typing.Any, ...]:
    return tuple(sorted(set(labels)))


def _get_box_descriptors(
    *args: AtomType | MoleculeType,
) -> tuple[
    dict[DescriptorType, tuple[LabelType, ...]],
    dict[TopologicalType, int],
    int,
]:
    descriptors: dict[DescriptorType, list[LabelType]] = {
        t: [] for t in descriptor_types
    }
    extras: dict[TopologicalType, int] = {t: 0 for t in topological_types}
    extra_special = 0

    for arg in args:
        if isinstance(arg, MoleculeType):
            for t in descriptor_types:
                descriptors[t].extend(arg._get_labels(t))
            for t in topological_types:
                extras[t] = max(arg._get_extra_per_atom(t), extras[t])
            extra_special = max(
                arg._get_extra_special_per_atom(), extra_special
            )
        else:
            descriptors["atom"].append(arg)

    unique_descriptors = {
        t: _unique_tuple(descriptors[t]) for t in descriptor_types
    }
    return unique_descriptors, extras, extra_special
