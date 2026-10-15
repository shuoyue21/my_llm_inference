#!/usr/bin/env python3
"""Compatibility entrypoint for the S0 frozen-model measurement queue."""
from stage_runner import main

if __name__ == '__main__':
    main(default_stage='S0')
