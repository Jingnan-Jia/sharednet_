#!/bin/bash

jobs=$(squeue -u jjia --sort=+i | grep [00-60]:[00-60] | awk '{print $1}')
echo what
echo $jobs
