# CLAUDE.md

## Dependencies

- Individual project dependencies are managed by `flake.nix`.
- When working with any of the projects in this repository, add any new
  dependencies you need by editing `flake.nix`.
- If a subdirectory doesn't yet have a `.envrc`, you can add an environment to
  `flake.nix` and add an appropriate `.envrc` (e.g. `use flake ..#<env>`).
- Commands you run in a subdirectory will automatically pick up changes to
  `.envrc` and `flake.nix`.
- Every dev shell exports `$DIRENV_LIB`, pointing at `scripts/direnv-lib.sh`,
  which defines shared direnv helpers such as `layout_uv`. To use them, source
  it after `use flake`:

  ```sh
  use flake ..#myproject
  source "$DIRENV_LIB"
  layout uv
  ```

## Publishing files to the web

- On hw4.nelhage.com, the `web` symlink at top-level points into the filesystem root for https://nelhage.com/ . If you place files inside the `web/` path, they will be visible under https://nelhage.com/files/sbox/
- Use this for sharing artifacts with your user -- you can make plots or render images and give viewable web URL.
