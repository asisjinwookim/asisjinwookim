#!/bin/bash

echo "Parent PID: $$"

nohup bash -c 'echo "Child PID: $$"; sleep 20; echo "Child process (PID: $$) finished."' &

sleep 10

echo "Parent process (PID: $$) will exit now."
exit 0

