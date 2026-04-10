# Nelson's sandbox monorepo

I use this repository for small projects or experiments that aren't (yet) worth their own repository. Each folder is a separate, independent, mini-project.

Projects use `nix` via a shared `flake.nix` to install any dependencies; many projects further use Python, in which case they define their own Python environments using `uv`.
