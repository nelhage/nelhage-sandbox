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
          pythonBase = pkgs.mkShell (
            {
              packages = [
                pkgs.python3
                pkgs.uv
                pkgs.ruff
              ];
            }
            // (
              if pkgs.stdenv.isDarwin then
                { }
              else
                {
                  shellHook = ''
                    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib"
                  '';
                }
            )
          );
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
          verso2docset = pythonBase;
          regexle = pythonBase;
          mnist-subliminal = pythonBase;
        }
      );
    };
}
