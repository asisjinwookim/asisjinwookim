import pytest

from .consistent_hashing import ConsistentHashing


@pytest.fixture
def consistent_hashing():
    return ConsistentHashing()


def test_add_node(consistent_hashing):
    consistent_hashing.add_node("node1")
    expected_vnodes = consistent_hashing.vnodes_by_node
    assert len(consistent_hashing.nodes) == 1
    assert "node1" in consistent_hashing.nodes
    assert len(consistent_hashing.ring) == expected_vnodes


def test_node_addition_and_removal(consistent_hashing):
    # Initial state assertions
    assert len(consistent_hashing.nodes) == 0
    assert len(consistent_hashing.ring) == 0

    print("consistent_hashing.ring", consistent_hashing.ring)

    nodes = ["node1", "node2", "node3"]

    assert len(consistent_hashing.nodes) == 0

    # Add nodes and assert the state after each addition
    for idx, node in enumerate(nodes):
        consistent_hashing.add_node(node)
        assert len(consistent_hashing.nodes) == idx + 1
        assert len(consistent_hashing.ring) == consistent_hashing.vnodes_by_node * (
            idx + 1
        )

    # Remove nodes and assert the state after each removal
    for idx, node in enumerate(nodes):
        consistent_hashing.remove_node(node)
        assert len(consistent_hashing.nodes) == len(nodes) - (idx + 1)
        assert len(consistent_hashing.ring) == consistent_hashing.vnodes_by_node * (
            len(nodes) - (idx + 1)
        )


def test_get_node(consistent_hashing):
    consistent_hashing.add_node("node1")
    consistent_hashing.add_node("node2")
    node = consistent_hashing.get_node("my_key")
    assert node in consistent_hashing.nodes


def test_redistribute_keys(consistent_hashing, capsys):
    consistent_hashing.add_node("node1")
    consistent_hashing.add_node("node2")
    consistent_hashing.add_node("node3")
    consistent_hashing.redistribute_keys("node3")
    captured = capsys.readouterr()
    assert "Moving key" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
