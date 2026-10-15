"""Explicit, recorded GPU isolation policies; compute clients are never exempt."""
import json
import os

STRICT_GPU_POLICY = {
    'schema_version': 1,
    'approved_graphics_process_names': [],
    'idle_counter_mode': 'strict',
}

def normalize_policy(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ValueError('GPU policy must be a JSON object')
    allowed = set(STRICT_GPU_POLICY) | {'description', 'authorization'}
    if set(value) - allowed:
        raise ValueError('Unknown GPU policy fields: ' + str(sorted(set(value) - allowed)))
    policy = dict(STRICT_GPU_POLICY, **value)
    if type(policy['schema_version']) is not int or policy['schema_version'] != 1:
        raise ValueError('Unsupported GPU policy version')
    names = policy['approved_graphics_process_names']
    if not isinstance(names, list) or any(not isinstance(name, str) or not name.startswith('/') for name in names):
        raise ValueError('Approved graphics names must be exact absolute process names')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate approved graphics name')
    if policy['idle_counter_mode'] not in ['strict', 'observe_only_with_approved_graphics']:
        raise ValueError('Unknown idle counter mode')
    if policy['idle_counter_mode'] != 'strict' and not names:
        raise ValueError('Observational counters require an explicitly approved graphics process')
    return policy

def policy_from_environment():
    value = os.environ.get('LLM_GPU_POLICY_JSON')
    return normalize_policy(json.loads(value) if value else None)
