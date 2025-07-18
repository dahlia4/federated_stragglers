#!/bin/bash
#set -e
#cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <file_path> <repeat_count> <missing>"
    exit 1
fi

file_path=$1
repeat_count=$2

# Check the value of the missing_flag argument
case "$3" in
    no-missing)
        missing_flag="no-missing"
    ;;
    missing)
        missing_flag="missing"
    ;;
    corrected)
        missing_flag="corrected"
    ;;
    *)
        usage
        exit 1
    ;;
esac

# Check if db/ and logs/ are not empty, and if so exit
if [ "$(ls -A logs/)" ]; then
    echo "ERROR: logs/ must be empty before running"
    # Ask user if they want to delete db/ and logs/
    read -p "Delete logs/ and continue? [y/n] " -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    rm logs/*
fi

machine_count=0

if [ ! -f "$file_path" ]; then
    echo "Error: File '$file_path' not found."
    exit 1
fi

first_machine=0

for line in `cat $file_path`; do
    if [ "$machine_count" -eq 0 ]; then
        first_machine=$line
        echo "Starting server on $line, stdout redirected to logs/server.log"
        ssh $first_machine "cd ~/thesis && source thesis_new/bin/activate && python src/server.py -server-ip $first_machine -server-port 9987 -missing $missing_flag 1>logs/server.log 2>>logs/server.log &" &
        sleep 3  # Sleep for 3s to give the server enough time to start
    fi
    
    offset=($machine_count*$repeat_count)
    machine_count=$((machine_count + 1))
    
    ssh $line "cd ~/thesis && source thesis_new/bin/activate && ./src/run_client.sh $first_machine 9987 $line $offset $repeat_count" &
    
    # for i in $(seq 1 $repeat_count); do
    #     client_count=$((client_count + 1))
    #     log_file="device.log"
    #     echo "Starting client $client_count on $line, stdout redirected to $log_file"
    #     port=$((10000 + client_count))
    #     ssh $line "cd ~/thesis && source thesis_new/bin/activate && nohup python src/device.py -server-ip $first_machine -server-port 9987 -client-ip $line -client-port $port -device-id $client_count >> logs/device.log 2>>logs/device.log &" &
    #     sleep 1.2
    # done
done

# Enable CTRL+C to stop all background processes

trap 'for line in `cat $file_path`; do
    ssh $line "killall -9 python"
done' SIGINT SIGTERM

# Wait for all background processes to complete
wait
