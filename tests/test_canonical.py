from paper_agent.canonical import canonical_json, content_hash


def test_canonical_json_is_order_independent() -> None:
    left = {"name": "paper", "values": [3, 2, 1]}
    right = {"values": [3, 2, 1], "name": "paper"}

    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)


def test_canonical_json_uses_rfc8785_number_format() -> None:
    assert canonical_json({"value": 1.0}) == b'{"value":1}'
