#!/bin/bash
set -e

# Для монтирования волума с корректными правами
chown agent:agent /work

exec /usr/sbin/sshd -D -e