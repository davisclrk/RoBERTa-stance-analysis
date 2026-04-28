"""Load PHEME threads and reconstruct branch contexts per reply tweet."""

import json
import glob
import os
from pathlib import Path
from typing import Optional

from config import DATA_RAW, LABELS

# Maps en-scheme-annotations.json responsetype values to SDQC label indices
_REPLY_LABEL_MAP = {
    "agreed": 0,                        # support
    "disagreed": 1,                     # deny
    "appeal-for-more-information": 2,   # query
    "comment": 3,                       # comment
}

PHEME_EN_DIR = DATA_RAW / "pheme-rumour-scheme-dataset" / "threads" / "en"
ANNOTATIONS_FILE = DATA_RAW / "pheme-rumour-scheme-dataset" / "annotations" / "en-scheme-annotations.json"


def _read_tweet_text(path: str) -> str:
    with open(path) as f:
        d = json.load(f)
    return d.get("text", "").strip()


def _build_file_maps(base_dir: Path) -> tuple[dict, dict]:
    """Return (reaction_map, source_map): tweet_id -> file path."""
    reaction_files = glob.glob(str(base_dir / "*" / "*" / "reactions" / "*.json"))
    source_files = glob.glob(str(base_dir / "*" / "*" / "source-tweets" / "*.json"))
    reaction_map = {os.path.splitext(os.path.basename(p))[0]: p for p in reaction_files}
    source_map = {os.path.splitext(os.path.basename(p))[0]: p for p in source_files}
    return reaction_map, source_map


def _flatten_tree(node, parent_id: Optional[str], parent_map: dict) -> None:
    """Recursively walk structure.json and populate parent_map[child] = parent."""
    if not isinstance(node, dict):
        return
    for child_id, subtree in node.items():
        if parent_id is not None:
            parent_map[child_id] = parent_id
        _flatten_tree(subtree, child_id, parent_map)


def _build_parent_maps(base_dir: Path) -> dict[str, dict]:
    """Return thread_id -> {tweet_id: parent_tweet_id} for all threads."""
    thread_parents: dict[str, dict] = {}
    for struct_path in glob.glob(str(base_dir / "*" / "*" / "structure.json")):
        with open(struct_path) as f:
            tree = json.load(f)
        thread_id = list(tree.keys())[0]
        parent_map: dict = {}
        _flatten_tree(tree, None, parent_map)
        thread_parents[thread_id] = parent_map
    return thread_parents


def _get_ancestors(tweet_id: str, parent_map: dict) -> list[str]:
    """Return ordered ancestor IDs from root down to immediate parent.

    Cycle-guarded: a malformed structure.json with a parent cycle would
    otherwise loop forever. We bail as soon as we revisit a node.
    """
    path = []
    seen = {tweet_id}
    current = tweet_id
    while current in parent_map:
        current = parent_map[current]
        if current in seen:
            break
        seen.add(current)
        path.append(current)
    path.reverse()  # root first
    return path


def _load_annotations(path: Path) -> tuple[list, list]:
    """Parse newline-delimited JSON annotation file.

    Returns (reply_entries, source_entries) where each entry is a raw dict.
    """
    with open(path) as f:
        content = f.read()
    lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]
    entries = [json.loads(l) for l in lines]
    reply_entries = [e for e in entries if e["tweetid"] != e["threadid"]]
    source_entries = [e for e in entries if e["tweetid"] == e["threadid"]]
    return reply_entries, source_entries


def load_pheme_dataset(events: Optional[list[str]] = None) -> list[dict]:
    """Load PHEME en threads and return one example per annotated reply tweet.

    Each example:
        tweet_id     : str
        thread_id    : str
        event        : str
        text         : str   target tweet text
        branch_texts : list  ancestor texts [root, ..., immediate_parent]
        label        : int   0=support 1=deny 2=query 3=comment

    Args:
        events: if given, restrict to those event names (e.g. ['charliehebdo']).
                Defaults to all 8 English events.
    """
    reaction_map, source_map = _build_file_maps(PHEME_EN_DIR)
    thread_parents = _build_parent_maps(PHEME_EN_DIR)
    reply_entries, _ = _load_annotations(ANNOTATIONS_FILE)

    if events is not None:
        events_set = set(events)
        reply_entries = [e for e in reply_entries if e["event"] in events_set]

    examples = []
    skipped = 0
    missing_ancestors = 0

    for entry in reply_entries:
        tweet_id = entry["tweetid"]
        thread_id = entry["threadid"]
        event = entry["event"]
        raw_label = entry.get("responsetype-vs-source")

        if raw_label not in _REPLY_LABEL_MAP:
            skipped += 1
            continue

        label = _REPLY_LABEL_MAP[raw_label]

        # Load target tweet text
        if tweet_id not in reaction_map:
            skipped += 1
            continue
        text = _read_tweet_text(reaction_map[tweet_id])

        # Trace ancestors and load their texts
        parent_map = thread_parents.get(thread_id, {})
        ancestor_ids = _get_ancestors(tweet_id, parent_map)

        branch_texts = []
        for anc_id in ancestor_ids:
            if anc_id in source_map:
                branch_texts.append(_read_tweet_text(source_map[anc_id]))
            elif anc_id in reaction_map:
                branch_texts.append(_read_tweet_text(reaction_map[anc_id]))
            else:
                # Rare: an intermediate tweet referenced by structure.json
                # has no JSON file. We continue with a shorter branch.
                missing_ancestors += 1

        examples.append({
            "tweet_id": tweet_id,
            "thread_id": thread_id,
            "event": event,
            "text": text,
            "branch_texts": branch_texts,
            "label": label,
        })

    if skipped:
        print(f"[data] skipped {skipped} entries (unknown label or missing file)")
    if missing_ancestors:
        print(f"[data] {missing_ancestors} ancestor tweet(s) had no JSON file — branch context truncated for those replies")

    return examples


def split_by_event(examples: list[dict]) -> dict[str, list[dict]]:
    """Group examples by event name for LOEO cross-validation."""
    splits: dict[str, list] = {}
    for ex in examples:
        splits.setdefault(ex["event"], []).append(ex)
    return splits


def loeo_splits(examples: list[dict]) -> list[tuple[str, list[dict], list[dict]]]:
    """Yield (held_out_event, train_examples, test_examples) for all events."""
    by_event = split_by_event(examples)
    all_events = list(by_event.keys())
    result = []
    for test_event in all_events:
        train = [ex for ev, exs in by_event.items() for ex in exs if ev != test_event]
        test = by_event[test_event]
        result.append((test_event, train, test))
    return result
