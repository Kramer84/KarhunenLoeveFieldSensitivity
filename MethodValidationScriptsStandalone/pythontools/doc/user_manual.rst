========================
Documentation of the API
========================

This is the user manual for the Python bindings to the quantile library.

.. currentmodule:: pythontools

Reliability Methods
===================

.. autosummary::
    :toctree: _generated/
    :template: class.rst_t
    :nosignatures:

    SystemEvent

.. autosummary::
    :toctree: _generated/
    :template: function.rst_t
    :nosignatures:

    run_monte_carlo
    run_importance_sampling
    run_subset
    run_directional
    run_FORM

Metamodel
=========

Kriging
-------

.. autosummary::
    :toctree: _generated/
    :template: function.rst_t
    :nosignatures:

    build_default_kriging_algo
    estimate_kriging_theta
    compute_LOO
    compute_Q2
