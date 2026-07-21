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
          # Android NDK for kindle-droid (cross-compiling the native hook .so).
          # Needs an unfree licence + accepted SDK licence, so import a config'd
          # pkgs instance just for this (the shared legacyPackages has neither).
          androidPkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              android_sdk.accept_license = true;
            };
          };
          androidNdk = (androidPkgs.androidenv.composeAndroidPackages {
            includeNDK = true;
          }).ndk-bundle;
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
          kindle-droid = pkgs.mkShell (
            {
              packages = [
                pkgs.python3
                pkgs.uv
                pkgs.ruff
                androidNdk
              ];
              ANDROID_NDK_ROOT = "${androidNdk}";
            }
            // claudeEnv
          );
          pypy = pkgs.mkShell (
            {
              packages = [
                pkgs.python3
                pkgs.pypy3
                pkgs.nodejs
                pkgs.librsvg
                pkgs.gcc
                pkgs.gnumake
              ];
            }
            // claudeEnv
          );
        }
      );
    };
}
