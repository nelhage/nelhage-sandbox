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

      withDirenvFor =
        pkgs:
        pkgs.runCommand "with-direnv"
          {
            nativeBuildInputs = [ pkgs.makeWrapper ];
          }
          ''
            install -Dm755 ${./scripts/with-direnv} $out/bin/with-direnv
            wrapProgram $out/bin/with-direnv \
              --prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.direnv ]}
          '';
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          with-direnv = withDirenvFor pkgs;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          withDirenv = withDirenvFor pkgs;
          claudeEnv = {
            CLAUDE_CODE_SHELL_PREFIX = "${withDirenv}/bin/with-direnv";
          };
          pythonBase = pkgs.mkShell (
            {
              packages = [
                pkgs.python3
                pkgs.uv
                pkgs.ruff
              ];
            }
            // claudeEnv
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
          default = pkgs.mkShell (
            {
              packages = [ ];
            }
            // claudeEnv
          );
          lean = pkgs.mkShell (
            {
              packages = [
                pkgs.elan
              ];
            }
            // claudeEnv
          );
          verso2docset = pythonBase;
          regexle = pythonBase;
          mnist-subliminal = pythonBase;
        }
      );
    };
}
