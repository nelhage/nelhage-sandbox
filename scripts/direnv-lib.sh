# -*- mode: sh; sh-shell: bash -*-
#
# direnv helpers shared by this repository. The dev shells in `flake.nix` export
# the store path of this file as $DIRENV_LIB; source it from `.envrc` (after
# `use flake`) to get these layouts without any personal dotfiles installed:
#
#     use flake ..#myproject
#     source "$DIRENV_LIB"
#     layout uv

layout_uv() {
    if [[ -d ".venv" ]]; then
        VIRTUAL_ENV="$(pwd)/.venv"
    fi

    if [[ -z $VIRTUAL_ENV || ! -d $VIRTUAL_ENV ]]; then
        log_status "No virtual environment exists. Executing \`uv venv\` to create one."
        uv venv
        VIRTUAL_ENV="$(pwd)/.venv"
    fi

    PATH_add "$VIRTUAL_ENV/bin"
    export UV_ACTIVE=1
    export VIRTUAL_ENV
}
