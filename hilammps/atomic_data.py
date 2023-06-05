from ase.data import atomic_masses, atomic_numbers

from hilammps.types import AtomType


def get_atomic_number(symbol: AtomType) -> int:
    """Get atomic number from symbol."""
    return atomic_numbers[symbol]


def get_atomic_mass(symbol: AtomType) -> float:
    """Get atomic mass from symbol."""
    return atomic_masses[get_atomic_number(symbol)]
