_reboot_prepend_lib_dir() {
    # Prepends $1 to LD_LIBRARY_PATH, skipping if already present or empty,
    # and never leaving a stray leading/trailing colon (which the dynamic
    # linker treats as "also search the current working directory").
    local dir="$1"
    [ -n "$dir" ] && [ -d "$dir" ] || return 0
    case ":${LD_LIBRARY_PATH}:" in
        *":${dir}:"*) return 0 ;;
    esac
    if [ -n "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${dir}"
    fi
}

init_reboot_experiment_paths() {
    local script_dir repo_root reboot_py_path rpath_dir

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export EXP_ROOT=$(realpath "${script_dir}/..")
    export EXP_DATA_DIR=$(realpath "${EXP_ROOT}/data")

    # reboot_py's baked-in RPATH doesn't always resolve OpenFHE's shared libs
    # at runtime (see doc/build.md) - add whichever OpenFHE this repo actually
    # built to LD_LIBRARY_PATH as a fallback, whether vendored or system.
    repo_root=$(realpath "${EXP_ROOT}/..")
    _reboot_prepend_lib_dir "${repo_root}/build/extern/openfhe-development-install/lib"

    if command -v readelf > /dev/null 2>&1 && command -v python3 > /dev/null 2>&1; then
        reboot_py_path=$(python3 -c "import reboot_py; print(reboot_py.__file__)" 2>/dev/null)
        if [ -n "$reboot_py_path" ] && [ -f "$reboot_py_path" ]; then
            rpath_dir=$(readelf -d "$reboot_py_path" 2>/dev/null \
                | grep -E "RPATH|RUNPATH" \
                | sed -E 's/.*\[(.*)\].*/\1/')
            _reboot_prepend_lib_dir "$rpath_dir"
        fi
    fi
}

init_reboot_experiment_paths
unset -f init_reboot_experiment_paths _reboot_prepend_lib_dir
