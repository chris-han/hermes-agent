import hashlib


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_leaf(event_hash: str) -> str:
    return _hash(("leaf:" + event_hash).encode())


def hash_node(left: str, right: str) -> str:
    return _hash(("node:" + left + right).encode())


def build_merkle_tree(event_hashes):
    if not event_hashes:
        return None, {}

    leaves = [hash_leaf(h) for h in event_hashes]
    tree = [leaves]

    while len(tree[-1]) > 1:
        level = tree[-1]
        next_level = []

        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(hash_node(left, right))

        tree.append(next_level)

    root = tree[-1][0]

    proof_map = {}
    for idx, leaf in enumerate(leaves):
        proof = []
        pos = idx

        for level in tree[:-1]:
            sibling_idx = pos ^ 1
            sibling = level[sibling_idx] if sibling_idx < len(level) else level[pos]

            direction = "left" if sibling_idx < pos else "right"

            proof.append((direction, sibling))
            pos //= 2

        proof_map[event_hashes[idx]] = proof

    return root, proof_map


def verify_proof(event_hash, proof, root):
    current = hash_leaf(event_hash)

    for direction, sibling in proof:
        if direction == "left":
            current = hash_node(sibling, current)
        else:
            current = hash_node(current, sibling)

    return current == root
