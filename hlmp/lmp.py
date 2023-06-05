# +
"""
This module provides a wrapper for the LAMMPS Python interface.

"""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Literal, Sequence

import numpy as np
from lammps import lammps as _lammps
from mpi4py import MPI

import hlmp.types as types
from hlmp.mpi import global_random_state
from hlmp.quant import QuanType, UnitType, lammps_factor

# Type aliases
IDType = Literal["region", "group", "fix", "compute", "molecule", "dump"]
CooType = tuple[float, float, float]


# A shallow wrapper for the LAMMPS Python interface
# to catch the commands and print them if verbose="yes"
class lammps(_lammps):
    # --- Hidden attributes ---
    _verbose: str = "no"
    _labels: dict[types.DescriptorType, dict[int, types.LabelType]] = {
        descriptor: {} for descriptor in types.descriptor_types
    }
    _cmd_count: int = 0

    # --- Commands ---

    def command(self, cmd: str) -> None:
        # Verbose output
        self._cmd_count += 1
        cmd = cmd.strip()
        if self._verbose != "no" and cmd != "":
            if self.get_mpi_comm().Get_rank() == 0:
                # if verbose is "yes", print command
                # otherwise write to a file with the name
                # specified by verbose
                if self._verbose == "yes":
                    print(f"NEOLAMMPS: {cmd}")
                else:
                    mode = "w" if self._cmd_count == 1 else "a"
                    with open(self._verbose, mode) as f:
                        f.write(f"{cmd}\n")

        if cmd.startswith("labelmap"):
            # If a labelmap is defined, store the info
            # since I don't know how to get it from LAMMPS
            # TODO: older versions of LAMMPS do not support labelmap
            self._parse_labelmap_command(f"{cmd}\n")
            if self.version() < 20230208:
                cmd = f"# {cmd}"

        # Execute the command
        super().command(cmd)

    def commands_list(self, cmds: list[str]) -> None:
        for cmd in cmds:
            self.command(cmd)

    def commands_string(self, cmds: str) -> None:
        if "&" in cmds:
            raise ValueError(
                "commands_string does not support multi-line commands yet"
            )
        self.commands_list(cmds.split("\n"))

    # --- Labels ---

    def _parse_labelmap_command(self, cmd: str) -> None:
        descriptor: types.DescriptorType
        words = cmd.strip().split()
        assert words[0] == "labelmap"
        if words[1] in ("clear", "write"):
            raise ValueError("labelmap clear/write is not supported yet")
        if words[1] not in types.descriptor_types:
            raise ValueError(f"Unknown labelmap option {words[1]}")
        # TODO: why type ignore?
        descriptor = words[1]  # type: ignore
        mapping = words[2:]
        assert len(mapping) % 2 == 0
        for i in range(0, len(mapping) // 2):
            type = int(mapping[2 * i])
            label = mapping[2 * i + 1]
            self._labels[descriptor][type] = label

    def get_label(
        self, descriptor: types.DescriptorType, type: int
    ) -> types.LabelType:
        return self._labels[descriptor][type]

    def get_type(
        self, descriptor: types.DescriptorType, label: types.LabelType
    ) -> int:
        for type, l in self._labels[descriptor].items():
            if l == label:
                return type
        raise ValueError(f"Label {label} not found in {descriptor}")


class Lammps:
    def __init__(
        self,
        comm: MPI.Intracomm | None = None,
        *,
        log: str | None = None,
        units: UnitType = "metal",
        seed: int | None = None,
        verbose: str = "no",
        screen: bool = False,
    ) -> None:
        """
        Initialize the Lammps class.

        Args:
            comm (MPI.Intracom | None, optional): MPI communicator.
            log (str | None, optional): log name.
            units (Literal["metal", "real", ...], optional): LAMMPS units.
            seed (int | None, optional): Random seed.
                seed is used to generate random seeds for internal LAMMPS
                commands that require one. The simulation should be
                reproducible if the same seed is used. If seed is None,
                a random seed is generated.
            verbose (str): Verbose mode.
                "no" will not print anything.
                "yes" will print the commands.
                Other strings will be used as a file name.
            screen (bool, optional): Screen mode.
                screen=True will print the LAMMPS output.
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        if log is None:
            log = "none"
        if screen:
            cmdargs = ["-log", log]
        else:
            cmdargs = ["-log", log, "-screen", "none"]
        self.lmp = lammps(cmdargs=cmdargs, comm=comm)
        self.lmp._verbose = verbose
        self.lmp.command(f"units {units}")
        if seed is None:
            self.seed = global_random_state.randint(2**32 - 1)
        else:
            self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

    def __del__(self) -> None:
        """Delete the NeoLammps class."""
        self.lmp.close()

    # --- Getters ---

    def get_randint(self, max: int = 2**16 - 1) -> int:
        """
        Get a random integer.
        This is to make sure random numbers are the same across all ranks.
        It also makes the simulation reproducible (with the same seed).
        This is mainly used for generating other random seeds whenever
        a seed argument is available in a LAMMPS command.
        Therefore the entire simulation should be reproducible with a
        single seed upon instant creation.

        Args:
            max (int, optional): Maximum value of the random integer.
                Note: if the max value is too large, using the random
                integer in LAMMPS commands may cause an overflow,
                segfault, or even worse silent errors.

        Returns:
            int: A random integer.
        """
        return self.random_state.randint(max)

    def get_units(self) -> UnitType:
        """
        Get units.

        Returns:
            LAMMPS units.
        """
        return self.lmp.extract_global("units")

    def get_factor(self, q: QuanType) -> float:
        """
        Get the conversion factor from ASE to internal LAMMPS units.

        Args:
            q: "distance", "time", "energy", etc.
        """
        units = self.lmp.extract_global("units")
        return lammps_factor(q, units)

    def get_natoms(self) -> int:
        """
        Get the number of atoms.

        Returns:
            int: Number of atoms.
        """
        return self.lmp.extract_global("natoms")

    # --- Setters ---

    def set_boundary(
        self, boundary: tuple[str | bool, str | bool, str | bool]
    ) -> None:
        """
        Set the boundary conditions.

        Args:
            boundary: Boundary conditions.
                True or "p" for periodic,
                False or "f" for fixed,
                "s" for shrink-wrapped,
                "m" for shrink-wrapped with minimum image convention.
        """

        def convert(x: bool | str) -> str:
            if x is True:
                return "p"
            elif x is False:
                return "f"
            else:
                assert x in (
                    "p",
                    "f",
                    "s",
                    "m",
                ), f"Invalid boundary condition: {x}"
                return x

        bx, by, bz = (convert(x) for x in boundary)
        self.lmp.command(f"boundary {bx} {by} {bz}")

    def set_timestep(self, timestep: float) -> None:
        """
        Set the timestep.

        Args:
            timestep (float): Timestep.

        Units:
            - time: ASE time unit.
        """
        c = self.get_factor("time")
        self.lmp.command(f"timestep {c * timestep}")

    # --- Ids and groups ---

    def automatic_id(
        self, category: IDType, *, suffix: str | None = None
    ) -> str:
        """
        Generate a new ID.

        Args:
            category (IDType): ID category ("region", "group", "fix", ...).
            suffix (str, optional): Suffix. Defaults to None.

        Returns:
            str: ID.
        """
        used_ids = self.lmp.available_ids(category)
        prefix: str
        if suffix is None:
            prefix = category
        else:
            prefix = f"{category}_{suffix}"
        for i in itertools.count(start=1):
            id = f"{prefix}_{i}"
            if id not in used_ids:
                break
            if i > 1000000:
                raise RuntimeError("Could not generate a new ID.")
        return id

    def group_range(self, i1: int, i2: int, *, id: str | None = None) -> Group:
        """
        Create a group.

        Args:
            i1 (int): ID of the first atom.
            i2 (int): ID of the last atom.
            id (str, optional): ID of the group. Defaults to None.
                If None, a new ID is generated automatically.

        Returns:
            Group: Group object.

        Note:
            Indices start from 0 (Python-like).
            They are internally converted to 1-based indices which
            are used in LAMMPS.
        """
        if id is None:
            id = self.automatic_id("group")
        self.lmp.command(f"group {id} id {i1+1}:{i2+1}")
        return Group(self.lmp, id)

    def group_atoms(
        self, *types: types.AtomType, id: str | None = None
    ) -> Group:
        """
        Create a group.

        Args:
            types (int | str): Atom types.
            id (str, optional): ID of the group. Defaults to None.
                If None, a new ID is generated automatically.

        Returns:
            Group: Group object.
        """
        if id is None:
            id = self.automatic_id("group")
        _types = (
            t if isinstance(t, int) else self.lmp.get_type("atom", t)
            for t in types
        )
        self.lmp.command(f"group {id} type {' '.join(str(t) for t in _types)}")
        return Group(self.lmp, id)

    def group_empty(self, *, id: str | None = None) -> Group:
        """
        Create an empty group.

        Args:
            id (str, optional): ID of the group. Defaults to None.
                If None, a new ID is generated automatically.

        Returns:
            Group: Group object.
        """
        if id is None:
            id = self.automatic_id("group")
        self.lmp.command(f"group {id} empty")
        return Group(self.lmp, id)

    def group_all(self) -> Group:
        """
        Return a handle for group containing all atoms.

        Returns:
            Group: Group object.
        """
        return Group(self.lmp, "all")

    # --- Region ---

    def region(
        self,
        high: CooType,
        *,
        low: CooType | None = None,
        tilt: CooType | None = None,
        id: str | None = None,
    ) -> Region:
        """
        Create a block or prism region.

        Args:
            high (CooType): High corner of the region.
            low (CooType, optional): Low corner of the region.
                None implies (0, 0, 0).
            tilt (CooType, optional): Tilt factors.
                If None, a block region is created.
                Enforcing a prism region with a orthogonal box can
                be done by specifying the tilt factors as (0, 0, 0).
            id (str, optional): ID of the region.
                If None, a new ID is generated automatically.

        Returns:
            Region: Region object.
        """
        if id is None:
            id = self.automatic_id("region")
        c = self.get_factor("distance")
        if low is None:
            xlo, ylo, zlo = 0.0, 0.0, 0.0
        else:
            xlo, ylo, zlo = tuple(c * x for x in low)
        xhi, yhi, zhi = tuple(c * x for x in high)
        if tilt is None:
            self.lmp.command(
                f"region {id} block"
                f" {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}"
                f" units box"
            )
            return Region(self.lmp, id)
        else:
            xy, xz, yz = tuple(c * x for x in tilt)
            self.lmp.command(
                f"region {id} prism "
                f"{xlo} {xhi} {ylo} {yhi} {zlo} {zhi} "
                f"{xy} {xz} {yz} units box"
            )
            return Region(self.lmp, id)

    def region_plane(
        self, point: CooType, normal: CooType, *, id: str | None = None
    ) -> Region:
        """
        Create a plane region.

        Args:
            point (CooType): Point on the plane.
            normal (CooType): Normal vector of the plane.
            id (str, optional): ID of the region.
                If None, a new ID is generated automatically.

        Returns:
            Region: Region object.
        """
        if id is None:
            id = self.automatic_id("region")
        c = self.get_factor("distance")
        x, y, z = tuple(c * x for x in point)
        nx, ny, nz = normal  # no need for: tuple(c * x for x in normal)
        self.lmp.command(f"region {id} plane {x} {y} {z} {nx} {ny} {nz}")
        return Region(self.lmp, id)

    def region_union(
        self, regions: Sequence[Region], *, id: str | None = None
    ) -> Region:
        """
        Create a union region.

        Args:
            regions (Sequence[Region]): Regions to be united.
            id (str, optional): ID of the region.
                If None, a new ID is generated automatically.

        Returns:
            Region: Region object.
        """
        if id is None:
            id = self.automatic_id("region")
        self.lmp.command(
            f"region {id} union {len(regions)} "
            f"{' '.join(r.id for r in regions)}"
        )
        return Region(self.lmp, id)

    # --- Box ---

    def create_box(
        self, region: Region, *ctx: types.AtomType | types.MoleculeType
    ) -> None:
        """
        Create a box.

        Args:
            region (Region): Region object.
            *ctx (AtomType | MoleculeType): Atom or molecule types.

        Notes:
            *ctx is used to infer the information needed for creating
            the box in LAMMPS. The following information is inferred:
                - number of atom types and their labels
                - number of bond types and their labels
                - number of angle types and their labels
                - number of dihedral types and their labels
                - number of improper types and their labels
                - number of extra per-atom properties
                - number of extra per-atom special properties

        Side effects:
            A labelmap is created for each type of labels.

        """

        # Create box
        descriptors, extras, extra_special = types._get_box_descriptors(*ctx)
        assert "atom" in descriptors
        atom_types = len(descriptors["atom"])
        command = f"create_box {atom_types} {region.id}"
        for descriptor, labels in descriptors.items():
            count = len(labels)
            if descriptor == "atom" or count == 0:
                continue
            command += f" {descriptor}/types {count}"
        for descriptor, extra in extras.items():
            if extra > 0:
                command += f" extra_{descriptor}_per_atom {extra}"
        if extra_special > 0:
            command += f" extra_special_per_atom {extra_special}"
        self.lmp.command(command)

        # Labelmap
        for descriptor, labels in descriptors.items():
            labelmap = " ".join(
                f"{i+1} {label}" for i, label in enumerate(labels)
            )
            self.lmp.command(f"labelmap {descriptor} {labelmap}")

    # --- Creations ---

    def create_atoms(
        self, positions: np.ndarray, labels: Sequence[types.AtomType]
    ) -> int:
        """
        Create atoms.

        Args:
            positions (np.ndarray): Positions of atoms.
            labels (Sequence[AtomType]): Atom types.

        Returns:
            int: number of atoms created.

        Units:
            Positions are in ASE distance units (Angstrom).
        """
        assert positions.shape[1] == 3
        assert positions.shape[0] == len(labels)
        c = self.get_factor("distance")
        if c != 1.0:
            positions = c * positions
        types = [self.lmp.get_type("atom", label) for label in labels]
        n = self.lmp.create_atoms(
            n=len(types),
            id=None,
            type=types,
            x=positions.flatten().tolist(),
            v=None,
            image=None,
            shrinkexceed=None,
        )
        count = Counter(types)
        self.lmp.command(f"# {n} atoms created from {count} types")
        return n

    def create_random_atoms(
        self,
        nmax: int,
        label: types.AtomType,
        overlap: float,
        *,
        region: Region | None = None,
    ) -> int:
        """
        Create atoms randomly.

        Args:
            nmax (int): Maximum number of atoms to create.
            label (AtomType): Atom type.
            overlap (float): Overlap tolerance.
            region (Region, optional): Region object.
                None means the whole simulation box.

        Returns:
            int: Number of new atoms created.

        Units:
            overlap is in ASE distance units (Angstrom).
        """
        seed = self.get_randint()
        region_id = region.id if region is not None else "NULL"
        n1 = self.lmp.extract_global("natoms")
        c = self.get_factor("distance")
        type = self.lmp.get_type("atom", label)
        self.lmp.command(
            f"create_atoms {type} "
            f"random {nmax} {seed} {region_id} "
            f"overlap {c * overlap}"
        )
        n2 = self.lmp.extract_global("natoms")
        n = n2 - n1
        self.lmp.command(f"# {n} atoms created from {type}")
        return n

    def create_random_velocities(
        self,
        temp: float,
        dist: Literal["gaussian", "uniform"] = "gaussian",
        *,
        group: Group | None = None,
    ) -> None:
        """
        Set random velocities.

        Args:
            temp (float): Temperature.
            dist (Literal["gaussian", "uniform"], optional): Distribution.
            group (Group, optional): Group of atoms.

        Units:
            temp is in ASE temperature units (Kelvin).

        Notes:
            The following keywords are implied:
                sum no
                mom yes
                rot yes
        """
        group_id = group.id if group is not None else "all"
        kwargs = "sum no mom yes rot yes"
        self.lmp.command(
            f"velocity {group_id} create {temp} "
            f"{self.get_randint()} dist {dist} {kwargs}"
        )


# --- Handles ---


class Group:
    def __init__(self, lmp: lammps, id: str) -> None:
        self.lmp = lmp
        self.id = id


class Region:
    def __init__(self, lmp: lammps, id: str) -> None:
        self.lmp = lmp
        self.id = id
