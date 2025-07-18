#!/bin/bash
#set -e
#cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

missing_flag=""

# Function to display usage message
usage() {
    echo "Usage: $0 <missing_flag>"
    echo "missing_flag options: no-missing, missing, corrected"
}

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

# Check the value of the missing_flag argument
case "$1" in
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

echo "Starting server"
python src/server.py -server-ip 127.0.0.1 -server-port 9987 -missing $missing_flag  > logs/server.log &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in {1..800}; do
    log_file="device${i}.log"
    echo "Starting client $i, stdout redirected to $log_file"
    port=$((10000 + i))
    python src/device.py -server-ip 127.0.0.1 -server-port 9987 -client-ip 127.0.0.1 -client-port "$port" -device-id "$i"  >> "logs/device.log" 2>&1 &
    sleep 0.01
done

# Enable CTRL+C to stop all background processes
trap 'pkill -f "python src/server.py" && pkill -f "python src/device.py"' SIGINT SIGTERM

# Wait for all background processes to complete
wait
