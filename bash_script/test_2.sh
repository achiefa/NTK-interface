#!/opt/homebrew/Cellar/bash/5.2.26/bin

set -e -o pipefail
#out/output_"${1%".txt"}
# shellcheck source=concurrent.lib.sh
source "/Users/s2569857/Codes/bash-concurrent/concurrent.lib.sh"
PATH_TO_SCRIPT="/Users/s2569857/Codes/NTK-interface/build/run"
TRAIN="${PATH_TO_SCRIPT}/train"
CONCURRENT_LIMIT=10

_log() {
        echo " $(date +'[%F %T]') - $1"
}


main() {
    # Starting the timer
    T1=$(date +%s)

    local args=()
    local i

    for (( i = 1; i <= $1; i++ )); do
        if [ $i -eq 1 ]
        then
            args+=(- "Replica ${i}" "./train" $1 "../../runcards/InputCard.yaml" "test" "./test/InputCard/Data.yaml" --and-then)
        else
            args+=(- "Replica ${i}" "./train" $1 "../../runcards/InputCard.yaml" "test" "./test/InputCard/Data.yaml")
        fi
    done

    concurrent "${args[@]}"
    
    # Ending the timer
    T2=$(date +%s)

    _log "All jobs completed in $((T2-T1)) seconds. Parent process exiting."
}

main "${@}"