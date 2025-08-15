{
  description = "direnv development template";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-linux"
        "x86_64-linux"
        "aarch64-darwin"
      ];
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [ ];
          };
          lean = pkgs.mkShell {
            packages = [
              pkgs.elan
            ];
          };
          verso2docset = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
              pkgs.ruff
            ];
          };
          regexle = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
              pkgs.ruff
            ];
          };
          mnist-subliminal = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
              pkgs.ruff
            ];
          };
        }
      );
    };
}
