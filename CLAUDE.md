# CLAUDE.md

## Dependencies
- Individual project dependencies are managed by `flake.nix`.
- When working with any of the projects in this repository, add any new
  dependencies you need by editing `flake.nix`.
- If a subdirectory doesn't yet have a `.envrc`, you can add an environment to
  `flake.nix` and add an appropriate `.envrc` (e.g. `use flake ..#<env>`).
- Commands you run in a subdirectory will automatically pick up changes to
  `.envrc` and `flake.nix`.
- Python projects should use `uv`, and track dependencies in a `pyproject.toml` and `layout uv` in the `.envrc` -- see `regexle` for an example. `layout uv` is configured on all of my machines, so you can assume it exists.

## General instructions
- Commit your work frequently. New projects will generally work on their own branch and you should commit after each tested unit of work.
- Many projects in this repository fetch or analyze datasets from elsewhere; downloaded or external datasets should generally live in `.gitignore` or be fetched to `~/.cache/`. They should NOT be committed.

## Publishing files to the web
- On hw4.nelhage.com, the `web` symlink at top-level points into the filesystem root for https://nelhage.com/ . If you place files inside the `web/` path, they will be visible under https://nelhage.com/files/sbox/
- Use this for sharing artifacts with your user -- you can make plots or render images and give viewable web URL.
