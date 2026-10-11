# SFILES 2.0 string fixtures

One canonical string per `.sfiles` file. Sources:

- `branch_recycle.sfiles`, `complex_hex_cycles.sfiles` — from upstream
  `SFILES2/run_demonstration.py` (MIT). Both round-trip exactly through
  `Flowsheet.create_from_sfiles` → `convert_to_sfiles(version="v2")`.

Add a control-notation fixture in WS0 step 3 once `sfiles.py` exists.
