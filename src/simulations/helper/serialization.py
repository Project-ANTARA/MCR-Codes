"""Load trusted pre-reorganization pickles without flat-module wrappers.

This is a name migration, not a security boundary. Never load untrusted pickle.
"""
import pickle


LEGACY_MODULES = {
    "antara_models": "simulations.helper.antara_models",
    "ANTARA_Science_Sim": "simulations.ditl.ANTARA_Science_Sim",
    "leop_detumble_mc": "simulations.detumble.leop_detumble_mc",
    "ANTARA_Master": "simulations.legacy.ANTARA_Master",
}


class RelocatedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(LEGACY_MODULES.get(module, module), name)


def load_trusted_pickle(stream):
    """Resolve historical module names when reading a trusted binary stream."""
    return RelocatedUnpickler(stream).load()
