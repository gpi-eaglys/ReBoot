_reboot_prepend_lib_dir() {
    # Prepends $1 to LD_LIBRARY_PATH, skipping if already present, and never
    # leaving a stray leading/trailing colon (which the dynamic linker treats
    # as "also search the current working directory"). Returns 1 (without
    # warning) if $1 isn't an existing directory, so callers can tell whether
    # anything useful was actually added.
    local dir="$1"
    [ -n "$dir" ] && [ -d "$dir" ] || return 1
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${dir}:"*) return 0 ;;
    esac
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${dir}"
    fi
}

_reboot_add_reboot_py_rpath() {
    # reboot_py's baked-in RPATH doesn't always get resolved automatically at
    # runtime (see doc/build.md) - locate its compiled .so and add every
    # directory listed in its RPATH/RUNPATH to LD_LIBRARY_PATH as a fallback.
    #
    # Uses importlib.util.find_spec() rather than `import reboot_py` on
    # purpose: an actual import would dlopen() the extension, which is
    # exactly what fails when OpenFHE's shared libs aren't found - that would
    # make this fallback unable to even locate the file in the one case it
    # exists to fix. find_spec() only locates the file, never loads it.
    if ! command -v python3 > /dev/null 2>&1; then
        echo "warning: paths.sh: python3 not found - skipping reboot_py RPATH fallback (OpenFHE shared libs may not be found at runtime)" >&2
        return 1
    fi
    if ! command -v readelf > /dev/null 2>&1; then
        echo "warning: paths.sh: readelf not found - skipping reboot_py RPATH fallback (OpenFHE shared libs may not be found at runtime)" >&2
        return 1
    fi

    local reboot_py_path
    reboot_py_path=$(python3 -c '
import importlib.util
spec = importlib.util.find_spec("reboot_py")
print(spec.origin if spec and spec.origin else "")
' 2>/dev/null)

    if [ -z "$reboot_py_path" ] || [ ! -f "$reboot_py_path" ]; then
        echo "warning: paths.sh: could not locate reboot_py's compiled module (is the venv active and 'reboot' installed?) - skipping RPATH fallback" >&2
        return 1
    fi

    local rpath_entries dir found=1
    # `|| true`: under a caller's `set -o pipefail`, grep finding no
    # RPATH/RUNPATH line would otherwise make this whole assignment "fail".
    rpath_entries=$( (readelf -d "$reboot_py_path" 2>/dev/null | grep -E "RPATH|RUNPATH" | sed -E 's/.*\[(.*)\].*/\1/') || true)

    if [ -z "$rpath_entries" ]; then
        echo "warning: paths.sh: ${reboot_py_path} has no RPATH/RUNPATH entry - OpenFHE shared libs may not be found at runtime" >&2
        return 1
    fi

    # rpath_entries may hold multiple lines (RPATH and RUNPATH both present)
    # and each line may itself be a colon-separated list of directories.
    local IFS=$' \t\n:'
    for dir in $rpath_entries; do
        _reboot_prepend_lib_dir "$dir" && found=0
    done

    if [ "$found" -ne 0 ]; then
        echo "warning: paths.sh: none of reboot_py's RPATH/RUNPATH directories exist on disk (${rpath_entries//$'\n'/, }) - OpenFHE shared libs may not be found at runtime" >&2
        return 1
    fi
}

init_reboot_experiment_paths() {
    local script_dir repo_root

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export EXP_ROOT=$(realpath "${script_dir}/..")
    export EXP_DATA_DIR=$(realpath "${EXP_ROOT}/data")
    export EXP_CONF_DIR=$(realpath "${EXP_ROOT}/config")

    # reboot_py's baked-in RPATH doesn't always resolve OpenFHE's shared libs
    # at runtime (see doc/build.md) - add whichever OpenFHE this repo actually
    # built to LD_LIBRARY_PATH as a fallback, whether vendored or system.
    # Both of these legitimately return 1 for "doesn't apply here" cases (e.g.
    # no vendored build on this machine) - `|| true` keeps that from being
    # mistaken for a real error by callers running under `set -e`.
    repo_root=$(realpath "${EXP_ROOT}/..")
    _reboot_prepend_lib_dir "${repo_root}/build/extern/openfhe-development-install/lib" || true

    _reboot_add_reboot_py_rpath || true
}

init_reboot_experiment_paths
unset -f init_reboot_experiment_paths _reboot_prepend_lib_dir _reboot_add_reboot_py_rpath
