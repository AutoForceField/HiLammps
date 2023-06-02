# +
"""
This module provides classes for representing
physical quantities and their automatic conversion
to the default units or other popular units systems.
The default units in HiLammps package are the same
as in ASE.

It is strongly recommended to use the classes
defined in this module for representing physical
quantities in this package. Aside from
readability, this will ensure that the units
are consistent and that the conversion between
different units is done automatically.

Quantities with physical units defined in LAMMPS
which ultimately should be covered by this module
are:
    mass, distance, time, energy, velocity,
    force, torque, temperature, pressure,
    dynamic_viscosity, charge, dipole,
    electric_field, density

Quantities are represented by classes inheriting
from the Quantity class. A few examples are:
    Time, Distance, Energy, etc.
The Quantity class is a subclass of float and
therefore can be used as a float. Its value is
the input value converted to the default unit.
For example:
    >>> from hilammps.quantity import Distance
    >>> d = Distance(1.0, "nm")
    >>> isinstance(d, float)
    True
    >>> print(d)
    10.0
    >>> print(d + 1.0)
    11.0
    >>> print(d + Distance(1.0, "A"))
    11.0
    >>> print(d + Distance(1.0, "nm"))
    20.0
    >>> print(10*d)
    100.0

In addition, one can use the get_value() method
to get the value in a specific unit system.
For example:
    >>> print(d.get_value("A"))
    10.0
    >>> print(d.get_value("ase"))
    10.0
    >>> print(d.get_value("real")) # LAMMPS real units
    10.0
    >>> print(d.get_value("si")) # LAMMPS si units
    1e-09

It can also be used in f-strings:
    >>> print(f"The distance is {d}")
    The distance is 10.0

"""
from __future__ import annotations

import abc

import ase.units as _units
from ase.calculators.lammps import convert

Number = float
LAMMPS_UNIT_SYSTEMS = (
    "real",
    "metal",
    "si",
    "cgs",
    "electron",
    "micro",
    "nano",
)
# Note, we have ignored "lj" units


class _Float(float):
    def __new__(cls, value: Number, unit: str | None = None) -> _Float:
        self = super().__new__(cls, value)
        return self


class Quantity(abc.ABC, _Float):
    """
    This is a class for representing physical
    values and their automatic conversions to
    a few popular units systems.
    The main interface method is
        get_value(units)
    where it should at least support the
    following units
        None (default)
        "ase"
        "lammps_real"
        "lammps_metal"
    Additional options are possible depending
    on the specific "Quantity" e.g "fs" for
    "Time".
    """

    @abc.abstractmethod
    def get_value(self, unit: str | None = None) -> Number:
        """
        unit:
            a specific unit or unit system like "ase",
            "lammps_real", etc. "None" means use the
            default unit.
        """
        ...

    @property
    def _name(self) -> str:
        return self.__class__.__name__

    _repr_unit: str

    def __repr__(self) -> str:
        u = self._repr_unit
        return f"{self._name}({self.get_value(u)}, '{u}')"

    def __str__(self) -> str:
        # __str__ returns the default unit
        return f"{self.get_value()}"

    def __float__(self) -> float:
        return float(self.get_value())

    def __add__(self, other: Number | Quantity) -> Number:
        return self.get_value() + other

    def __radd__(self, other: Number | Quantity) -> Number:
        return other + self.get_value()

    def __sub__(self, other: Number | Quantity) -> Number:
        return self.get_value() - other

    def __rsub__(self, other: Number | Quantity) -> Number:
        return other - self.get_value()

    def __mul__(self, other: Number | Quantity) -> Number:
        return self.get_value() * other

    def __rmul__(self, other: Number | Quantity) -> Number:
        return other * self.get_value()

    def __truediv__(self, other: Number | Quantity) -> Number:
        return self.get_value() / other

    def __rtruediv__(self, other: Number | Quantity) -> Number:
        return other / self.get_value()

    def __pow__(self, other: Number | Quantity) -> Number:  # type: ignore
        return self.get_value() ** other

    def __rpow__(self, other: Number | Quantity) -> Number:  # type: ignore
        return other ** self.get_value()

    def __neg__(self) -> Number:
        return -self.get_value()

    def __pos__(self) -> Number:
        return +self.get_value()

    def __abs__(self) -> Number:
        return abs(self.get_value())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Quantity):
            return self.get_value() == other.get_value()
        elif isinstance(other, Number):
            return self.get_value() == other
        else:
            return NotImplemented

    def __ne__(self, other: object) -> bool:
        if isinstance(other, Quantity):
            return self.get_value() != other.get_value()
        elif isinstance(other, Number):
            return self.get_value() != other
        else:
            return NotImplemented


class UnitNotFound(Exception):
    def __init__(self, quantity: Quantity, unit: str) -> None:
        message = f"The unit '{unit}' is not defined for '{quantity._name}'!"
        super().__init__(message)


class _Convertable(Quantity):
    _units: dict[str, float | str]

    def __init__(self, value: Number = 1.0, unit: str | None = None) -> None:
        """
        If unit == None, default unit is implied.
        """
        self._value = value * self._get_coef(unit)
        if unit is not None:
            self._repr_unit = unit

    def get_value(self, unit: str | None = None) -> Number:
        coef = self._get_coef(unit)
        if coef == 1.0:
            return self._value
        else:
            return self._value / coef

    def _get_coef(self, unit: str | None) -> float:
        if unit is None:
            return 1.0
        _unit = unit.lower()
        if unit in self._units:
            c = self._units[unit]
            if isinstance(c, str):
                return self._get_coef(c)
            else:
                return c
        elif _unit.startswith("lammps_") or _unit in LAMMPS_UNIT_SYSTEMS:
            assert "ase" in self._units
            q = self._name.lower()
            u = _unit.replace("lammps_", "")
            return convert(1, q, u, "ASE") / self._units["ase"]
        else:
            raise UnitNotFound(self, unit)

    @classmethod
    def available_units(cls) -> tuple[str, ...]:
        return tuple(cls._units.keys()) + LAMMPS_UNIT_SYSTEMS


class Time(_Convertable):
    """
    Defined units:
        fs: femtosecond (default)
        ps: picosecond
        ns: nanosecond
        s:  second

    Other unit systems:
        ase
        lammps_real
        lammps_metal
        etc.

    """

    _units: dict[str, float | str] = {
        "fs": 1.0 * _units.fs,
        "ps": 1e3 * _units.fs,
        "ns": 1e6 * _units.fs,
        "s": 1e15 * _units.fs,
        "ase": 1,
    }
    _repr_unit: str = "fs"


class Distance(_Convertable):
    """
    Defined units:
        A:  Angstrom (default)
        nm: nanometer
        m:  meter
        cm: centimeter

    Other unit systems:
        ase
        lammps_real
        lammps_metal
        etc.

    """

    _units: dict[str, float | str] = {
        "A": 1,
        "nm": 10.0,
        "cm": 1e8,
        "m": 1e10,
        "ase": 1,
    }
    _repr_unit: str = "A"


class Mass(_Convertable):
    """
    Defined units:
        amu: atomic mass unit (default)
        gr:  gram
        kg:  kilogram

    Other unit systems:
        ase
        lammps_real
        lammps_metal
        etc.

    """

    _units: dict[str, float | str] = {
        "amu": 1,
        "gr": _units.mol,
        "kg": 1e3 * _units.mol,
        "ase": 1,
    }
    _repr_unit: str = "amu"


_gr = Mass(1.0, "gr").get_value("amu")
_cm = Distance(1.0, "cm").get_value("A")
_gr_cm3: float = _gr / _cm**3  # type: ignore


class Density(_Convertable):
    """
    Defined units:
        amu/A3: atomic mass unit per cubic angstrom (default)
        gr/cm3: gram per cubic centimeter

    Other unit systems:
        ase
        lammps_real
        lammps_metal
        etc.

    """

    _units: dict[str, float | str] = {
        "amu/A3": 1,
        "gr/cm3": _gr_cm3,
        "ase": 1,
        # TODO: other units
    }
    _repr_unit: str = "amu/A3"

    def get_number_density(self, mass: float) -> Number:
        """
        mass:
            mass of the atom/molecule in amu
        """
        return self.get_value("amu/A3") / mass


class Energy(_Convertable):
    """
    Defined units:
        eV:  electronvolt (default)

    Other unit systems:
        ase
        lammps_real
        lammps_metal
        etc.

    """

    _units: dict[str, float | str] = {
        "eV": 1,
        "kcal": _units.kcal,
        "kcal/mol": _units.kcal / _units.mol,
        "kJ": _units.kJ,
        "kJ/mol": _units.kJ / _units.mol,
        "Joules": "si",
        "J": "Joules",
        "ergs": "cgs",
        "ase": 1,
    }
    _repr_unit: str = "eV"


class Pressure(_Convertable):
    _units: dict[str, float | str] = {
        "pa": _units.Pascal,
        "gpa": _units.GPa,
        "bar": _units.bar,
        "atm": convert(1, "pressure", "real", "ASE"),
        "ase": 1,
    }
    _repr_unit: str = "bar"


class Temperature(Quantity):
    """
    Defined units:
        k: Kelvin (default)
        c: Celsius

    Other unit systems:
        ase
        lammps_real
        lammps_metal
        etc.

    """

    _repr_unit: str = "k"

    def __init__(self, value: Number, unit: str | None = None) -> None:
        self._value = value + self._get_delta(unit)

    def get_value(self, unit: str | None = None) -> Number:
        return self._value - self._get_delta(unit)

    def _get_delta(self, unit: str | None = None) -> Number:
        if unit is None or unit == "k" or unit == "ase":
            return 0
        elif unit.startswith("lammps_"):
            u = unit.split("_")[1]
            if u in LAMMPS_UNIT_SYSTEMS:
                return 0
            else:
                raise UnitNotFound(self, unit)
        elif unit == "c":
            return 273.15
        else:
            raise UnitNotFound(self, unit)


def test_all_convertables() -> bool:
    for cls in _Convertable.__subclasses__():
        for u in cls._units:
            assert cls(1, u).get_value(u) == 1
    return True


def test_arithmetic_operations() -> bool:
    t = Time(1)
    assert t + 1 == Time(2)
    assert t + t == Time(2)
    assert t - t == Time(0)
    assert t * 2 == Time(2)
    assert 2 * t == Time(2)
    assert t / 2 == Time(0.5)
    assert t / t == 1
    assert t * t == 1
    # test pow
    t = Time(2)
    assert t**2 == Time(4)
    assert t**-1 == Time(0.5)
    assert t**0 == Time(1)
    assert t**t == Time(4)

    return True


def test_fsrings() -> bool:
    t = Time(1, "ns")
    assert float(f"{t}") == 1e6 * _units.fs
    return True


def test_typing() -> bool:
    def f(x: float) -> float:
        return x

    t = Time(1, "ns")
    assert f(t) == 1e6 * _units.fs
    return True


def test_special_Time() -> bool:
    for u in "fs ps ns s".split():
        assert Time(1, u).get_value(u) == 1
    t = Time(1.0, "ase")
    assert _are_close(t.get_value("fs"), t.get_value("lammps_real"))
    assert _are_close(t.get_value("ps"), t.get_value("lammps_metal"))

    return True


def test_special_Distance() -> bool:
    for u in "A nm cm m".split():
        assert Distance(1, u).get_value(u) == 1
    d = Distance(1.0, "ase")

    # test conversions between units
    assert _are_close(d.get_value("cm"), 1e2 * d.get_value("m"))
    assert _are_close(d.get_value("nm"), 1e-1 * d.get_value("A"))
    assert _are_close(d.get_value("m"), 1e-10 * d.get_value("A"))

    # lammps related
    assert _are_close(d.get_value("A"), d.get_value("lammps_real"))
    assert _are_close(d.get_value("A"), d.get_value("lammps_metal"))
    assert _are_close(d.get_value("m"), d.get_value("lammps_si"))
    assert _are_close(d.get_value("cm"), d.get_value("lammps_cgs"))

    return True


def test_special_Mass() -> bool:
    for u in "amu gr kg".split():
        assert Mass(1, u).get_value(u) == 1
    m = Mass(1.0, "ase")

    # test conversions between units
    assert _are_close(m.get_value("gr"), 1e3 * m.get_value("kg"))

    # lammps related
    assert _are_close(m.get_value("amu"), m.get_value("lammps_real"))
    assert _are_close(m.get_value("amu"), m.get_value("lammps_metal"))
    assert _are_close(m.get_value("kg"), m.get_value("lammps_si"))
    assert _are_close(m.get_value("gr"), m.get_value("lammps_cgs"))

    return True


def _are_close(a, b, rtol=1e-8):
    if isinstance(a, Quantity):
        a = a.get_value()
    if isinstance(b, Quantity):
        b = b.get_value()
    return abs(a - b) / min(abs(a), abs(b)) < rtol


if __name__ == "__main__":
    test_all_convertables()
    test_arithmetic_operations()
    test_fsrings()
    test_typing()
    test_special_Time()
    test_special_Distance()
    test_special_Mass()
