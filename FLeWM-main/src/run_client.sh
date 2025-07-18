#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <server_ip> <server_port> <local_ip> <device_offset> <num_clients>"
    exit 1
fi

server_ip="$1"
server_port="$2"
local_ip="$3"
device_offset="$4"
num_clients="$5"

for i in $(seq 1 "$num_clients"); do
    device_id=$((device_offset + i))
    log_file="device${i}.log"
    echo "Starting client $device_id"
    port=$((10000 + i))
    python src/device.py -server-ip "$server_ip" -server-port "$server_port" -client-ip "$local_ip" -client-port "$port" -device-id "$device_id" >> "logs/device.log" 2>&1 &
    sleep 1
done