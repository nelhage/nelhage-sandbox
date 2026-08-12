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

      # Shared direnv helpers (`layout uv`, ...). Every dev shell exports this
      # as $DIRENV_LIB so a `.envrc` can `source "$DIRENV_LIB"` instead of
      # relying on a personal ~/.config/direnv/direnvrc.
      direnvLib = ./scripts/direnv-lib.sh;
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
          inherit (pkgs) lib;
          withDirenv = withDirenvFor pkgs;

          # Every shell derives from `baseEnv`; `withPackages` layers extra
          # tools on top via overrideAttrs.
          baseEnv = pkgs.mkShell {
            name = "sandbox-base";
            packages = [ ];
            CLAUDE_CODE_SHELL_PREFIX = "${withDirenv}/bin/with-direnv";
            DIRENV_LIB = "${direnvLib}";
          };

          withPackages =
            extra: shell:
            shell.overrideAttrs (old: {
              nativeBuildInputs = old.nativeBuildInputs ++ extra;
            });

          pythonEnv =
            (withPackages [
              pkgs.python3
              pkgs.uv
              pkgs.ruff
            ] baseEnv).overrideAttrs
              (
                old:
                lib.optionalAttrs pkgs.stdenv.isLinux {
                  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
                }
              );

          # Android NDK
          # Needs an unfree licence + accepted SDK licence, so import a config'd
          # pkgs instance just for this (the shared legacyPackages has neither).
          androidPkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              android_sdk.accept_license = true;
            };
          };
          androidNdk =
            (androidPkgs.androidenv.composeAndroidPackages {
              includeNDK = true;
            }).ndk-bundle;

        in
        {
          default = baseEnv;

          lean = withPackages [ pkgs.elan ] baseEnv;

          verso2docset = pythonEnv;
          regexle = pythonEnv;
          mnist-subliminal = pythonEnv;

          # `pandoc` turns the generated Markdown into EPUB for e-readers.
          pdf2md = withPackages [ pkgs.pandoc ] pythonEnv;

          pypy = withPackages [
            pkgs.pypy3
            pkgs.nodejs
            pkgs.librsvg
            pkgs.gcc
            pkgs.gnumake
          ] pythonEnv;

          ff-profile =
            (withPackages [
              pkgs.samply
              androidNdk
            ] pythonEnv).overrideAttrs
              (old: {
                ANDROID_NDK_ROOT = "${androidNdk}";
              });
        }
      );
    };
}
