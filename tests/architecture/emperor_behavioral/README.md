# Emperor behavioral records

This directory contains the compact records that the Emperor behavioral
architecture gate is allowed to consume. Raw coverage databases and reports,
mutation caches, logs, review drafts, and other temporary investigation output
belong under `.scratch/` and are not authoritative inputs.

The family `status` in `../emperor_test_manifest.toml` is the certification
source of truth. An evidence file records one specific completed check; the
existence of that file does not by itself make a family complete. A `complete`
family must have passing coverage, mutation, and independent-review records
whose module lists match the exact modules registered for that family in the
manifest.

Evidence for a partial or pending family may remain here as historical progress
without representing current certification. Adding a production module requires
adding its meaningful `module-ledger.csv` row in the same change. Tests and the
authoritative manifest must never reference `.scratch/` files.
