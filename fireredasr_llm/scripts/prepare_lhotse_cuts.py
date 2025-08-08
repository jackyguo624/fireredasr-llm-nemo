#!/usr/bin/env python3
"""
Prepare lhotse cuts.jsonl files from recordings.jsonl and supervisions.jsonl
using the lhotse library
"""

import argparse
import os
from lhotse import RecordingSet, SupervisionSet, CutSet


def prepare_cuts_from_lhotse(recordings_path, supervisions_path, output_path):
    """Prepare cuts.jsonl.gz using lhotse library"""
    
    print(f"Loading recordings from: {recordings_path}")
    recordings = RecordingSet.from_file(recordings_path)
    
    print(f"Loading supervisions from: {supervisions_path}")
    supervisions = SupervisionSet.from_file(supervisions_path)
    
    print("Creating cuts...")
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    
    print(f"Writing cuts to: {output_path}")
    cuts.to_file(output_path)
    
    print(f"Created {len(cuts)} cuts")
    return len(cuts)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare lhotse cuts.jsonl.gz from recordings and supervisions files"
    )
    parser.add_argument(
        "--recordings", "-r",
        required=True,
        help="Path to recordings.jsonl.gz file"
    )
    parser.add_argument(
        "--supervisions", "-s",
        required=True,
        help="Path to supervisions.jsonl.gz file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for cuts.jsonl.gz file"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare cuts
    num_cuts = prepare_cuts_from_lhotse(args.recordings, args.supervisions, args.output)
    
    print(f"Successfully created {num_cuts} cuts at {args.output}")


if __name__ == "__main__":
    main()