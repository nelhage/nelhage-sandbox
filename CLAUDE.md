# CLAUDE.md

## Dependencies

- Individual project dependencies are managed by `flake.nix`.
- When working with any of the projects in this repository, add any new
  dependencies you need by editing `flake.nix`.
- If a subdirectory doesn't yet have a `.envrc`, you can add an environment to
  `flake.nix` and add an appropriate `.envrc` (e.g. `use flake ..#<env>`).
- Commands you run in a subdirectory will automatically pick up changes to
  `.envrc` and `flake.nix`.
