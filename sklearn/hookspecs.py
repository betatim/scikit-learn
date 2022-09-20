from pluggy import HookimplMarker
from pluggy import HookspecMarker

hookspec = HookspecMarker("scikit-learn")
hookimpl = HookimplMarker("scikit-learn")


# Specification of all the hooks that a plugin can implement


@hookspec(firstresult=True)
def kmeans_single_lloyd(
    X, sample_weight, centers_init, max_iter, verbose, tol, n_threads
):
    pass


@hookspec(firstresult=True)
def kmeans_single_elkan(
    X, sample_weight, centers_init, max_iter, verbose, tol, n_threads
):
    pass
