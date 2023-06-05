from hlmp.lmp import Lammps


def test_lmp():
    lmp = Lammps(verbose="yes")
    reg = lmp.region((10.0, 10.0, 10.0))
    lmp.create_box(reg, "A", "B")
    lmp.create_random_atoms(10, "A", 3.0)


if __name__ == "__main__":
    test_lmp()
