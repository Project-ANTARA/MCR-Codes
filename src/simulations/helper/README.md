# Helper simulations and utilities

This folder owns these implementations: `antara_models.py`, `fsm_spec.py`, `paths.py`, `serialization.py`, `slew_profile.py`, `telemetry_metrics.py`, `wheel_dynamics.py`, `pass_statistics.py`, `event_archive.py`.

## Documentation

- [antara_models](antara_models.md)
- [paths](paths.md)
- [Bounded payload pass statistics](../ditl/BOUNDED_PASS_STATISTICS.md)
- [Science event archives and checkpoint retention](../ditl/COMPLETED_PASS_ARCHIVE.md)

Run entrypoints from the repository root using their full `src/simulations/helper/` path. Shared imports use `simulations.helper`. Some studies execute on import; do not import them merely to inspect the layout.

[Repository layout](../../../docs/SIMULATION_LAYOUT.md)
