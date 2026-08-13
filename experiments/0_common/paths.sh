init_reboot_experiment_paths() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export EXP_ROOT=$(realpath "${script_dir}/..")
    export EXP_DATA_DIR=$(realpath "${EXP_ROOT}/data")
}

init_reboot_experiment_paths
unset -f init_reboot_experiment_paths

