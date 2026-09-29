# Generated artifacts

This directory is reserved for local generated files. The current legacy entry points
write search artifacts to `Search/` and content artifacts plus `manifest.json` to
`SIM/`. Large binary files are ignored by Git.

The content builder produces a schema 2 manifest with the ordered catalog checksum,
file checksums, model identity, and row counts. The audit checks both index row orders
against their embedding matrices and rejects zero or non-finite vectors. The desktop
app requires this manifest and will reject older artifacts until rebuilt or migrated.
