"""Validate an analysis_recipe.json against analysis_recipe.schema.json.

Usage:
    python scripts/validate_recipe.py analysis_recipe.json
    python scripts/validate_recipe.py --schema custom_schema.json analysis_recipe.json

Exit code 0 = valid, 1 = invalid or error.
This script should be run as a gate before passing a recipe to C++ inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Validate analysis_recipe.json against schema.')
    parser.add_argument('recipe', help='Path to analysis_recipe.json')
    parser.add_argument('--schema', default='',
                        help='Path to analysis_recipe.schema.json. '
                             'Defaults to <project_root>/analysis_recipe.schema.json')
    args = parser.parse_args()

    recipe_path = Path(args.recipe)
    if not recipe_path.is_file():
        print(f'ERROR: recipe not found: {recipe_path}')
        sys.exit(1)

    if args.schema:
        schema_path = Path(args.schema)
    else:
        # default: same directory as this script's parent (project root)
        script_dir = Path(__file__).resolve().parent
        schema_path = script_dir.parent / 'analysis_recipe.schema.json'

    if not schema_path.is_file():
        print(f'ERROR: schema not found: {schema_path}')
        sys.exit(1)

    try:
        with recipe_path.open('r', encoding='utf-8') as f:
            recipe = json.load(f)
    except json.JSONDecodeError as exc:
        print(f'ERROR: invalid JSON in {recipe_path}: {exc}')
        sys.exit(1)

    try:
        with schema_path.open('r', encoding='utf-8') as f:
            schema = json.load(f)
    except json.JSONDecodeError as exc:
        print(f'ERROR: invalid JSON in {schema_path}: {exc}')
        sys.exit(1)

    # Validate using jsonschema if available, otherwise do basic structural check
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(recipe), key=lambda e: e.path)
        if errors:
            print(f'ERROR: recipe failed schema validation ({len(errors)} issue(s)):')
            for err in errors:
                path_str = ' → '.join(str(p) for p in err.path) if err.path else '(root)'
                print(f'  {path_str}: {err.message}')
            sys.exit(1)
        print(f'OK: {recipe_path} is valid per {schema_path}')
    except ImportError:
        # Fallback: basic structural check without jsonschema
        print('WARNING: jsonschema not installed; using basic structural check only.')
        print('  Install with: pip install jsonschema')
        errors = _basic_check(recipe, schema)
        if errors:
            print(f'ERROR: recipe failed basic structural check ({len(errors)} issue(s)):')
            for err in errors:
                print(f'  {err}')
            sys.exit(1)
        print(f'OK: {recipe_path} passed basic structural check per {schema_path}')
        print('  (Install jsonschema for full JSON Schema validation)')


def _basic_check(instance: dict, schema: dict) -> list[str]:
    """Minimal structural check when jsonschema is not available."""
    errors: list[str] = []

    # Check required top-level fields
    for field in schema.get('required', []):
        if field not in instance:
            errors.append(f"missing required field: {field}")

    # Check enum constraints on known fields
    props = schema.get('properties', {})
    for field_name, field_schema in props.items():
        if field_name not in instance:
            continue
        if 'enum' in field_schema:
            value = instance[field_name]
            if value not in field_schema['enum']:
                errors.append(
                    f"{field_name}: '{value}' not in {field_schema['enum']}")

    # Check resolved_parameters sub-objects
    rp = instance.get('resolved_parameters', {})
    rp_props = props.get('resolved_parameters', {}).get('properties', {})
    for sub_name, sub_schema in rp_props.items():
        if sub_name not in rp:
            errors.append(f"resolved_parameters missing required sub-field: {sub_name}")
        elif sub_schema.get('required'):
            for req_field in sub_schema['required']:
                if req_field not in rp[sub_name]:
                    errors.append(
                        f"resolved_parameters.{sub_name} missing required field: {req_field}")

    # Check scale conditional: if has_scale=true, pixel_size etc required
    scale = instance.get('scale', {})
    if scale.get('has_scale') is True:
        scale_schema = props.get('scale', {})
        for then_req in ['pixel_size', 'pixel_size_unit', 'source']:
            if then_req not in scale:
                errors.append(f"scale.has_scale=true requires: {then_req}")

    return errors


if __name__ == '__main__':
    main()
