import importlib
import sys

import pluggy
from . import hookspecs


# Using a class to namespace the "cupy" plugin. In reality this would be
# implemented in a separate package like `scikit-learn-cupy`
class CuPY:
    @hookspecs.hookimpl(specname="kmeans_single_lloyd")
    def cupy_kmeans_single_lloyd(
        self, X, sample_weight, centers_init, max_iter, verbose, tol, n_threads
    ):
        # This implementation should be in a package like scikit-learn-cupy
        # where it can test if X is a cupy array, or use some other method
        # to decide if it wants to handle this call
        # For example it could check if X.shape is "big enough" for a GPU
        # transfer to be worth it, and only act if it is

        print("cupy lloyd called")
        # Here we hardcode the "we don't want to do anything" behaviour.
        # All hook implementations will be called in order until the first
        # one returns a non-None value.
        return None


manager = pluggy.PluginManager("scikit-learn")
manager.add_hookspecs(hookspecs)

# Register "base" plugins implemented in scikit-learn
for plugin in ("sklearn.cluster._kmeans",):
    module = importlib.import_module(plugin)
    manager.register(module, plugin)

# Register the fake "cupy" plugin
manager.register(CuPY(), "scikit-learn-cupy")
# In the real world: find external plugins via the entrypoints mechanism
# instead of explicitly calling `pm.register()`
# manager.load_setuptools_entrypoints(...)
