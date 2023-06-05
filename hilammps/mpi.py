# +
from __future__ import annotations

import abc
import io
import tempfile

import numpy as np
from mpi4py import MPI


class Communicator(abc.ABC):
    @abc.abstractmethod
    def Get_size(self) -> int:
        ...

    @abc.abstractmethod
    def Get_rank(self) -> int:
        ...

    @abc.abstractmethod
    def Barrier(self) -> None:
        ...


Communicator.register(MPI.Comm)
world = MPI.COMM_WORLD
is_master = world.Get_rank() == 0

# global random seed
if is_master:
    _global_random_seed = np.random.randint(2**16 - 1)
else:
    _global_random_seed = None
_global_random_seed = world.bcast(_global_random_seed, root=0)
global_random_state = np.random.RandomState(_global_random_seed)


def strio_to_file(
    strio: io.StringIO, file: "str" | io.TextIOWrapper | None, mode: str = "w"
) -> str:
    if world.Get_rank() == 0:
        if isinstance(file, io.TextIOWrapper):
            file.write(strio.getvalue())
            name = file.name
        elif type(file) == str:
            with open(file, mode) as of:
                of.write(strio.getvalue())
            name = file
        elif file is None:
            tmp = tempfile.NamedTemporaryFile(mode, suffix="_Wurtzite")
            tmp.write(strio.getvalue())
            tmp.flush()
            name = tmp.name
            # for avoiding tmp deletion, we store in a global list
            _tmpfiles.append(tmp)
        else:
            raise
    else:
        name = None
    name = world.bcast(name, root=0)
    return name  # type: ignore # mypy can't infer bcast


_tmpfiles: list[tempfile._TemporaryFileWrapper] = []
