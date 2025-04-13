from unittest.mock import patch

import pytest

from .raft import RaftNode


# Import the RaftNode class from the module where it is defined
# from your_module import RaftNode  # Uncomment and update if you use a separate module


@pytest.fixture
def setup_nodes():
    nodes = [RaftNode(i, []) for i in range(5)]
    for i in range(5):
        nodes[i].peers = nodes[:i] + nodes[i + 1 :]
    return nodes


def test_election_process(setup_nodes):
    nodes = setup_nodes

    with patch("time.sleep", return_value=None):  # Patch sleep to speed up the test
        nodes[0].start_election()

    # Check that node 0 became the leader
    assert nodes[0].state == "leader"
    # Check that other nodes are still followers
    for node in nodes[1:]:
        assert node.state == "follower"


def test_vote_request_and_grant(setup_nodes):
    nodes = setup_nodes

    # Node 0 starts an election
    nodes[0].start_election()

    # All other nodes should vote for node 0
    for node in nodes[1:]:
        assert node.voted_for == 0


def test_leader_append_entries(setup_nodes):
    nodes = setup_nodes

    with patch("time.sleep", return_value=None):  # Patch sleep to speed up the test
        nodes[0].start_election()

    # After becoming leader, node 0 should send heartbeats
    assert nodes[0].state == "leader"
    with patch("time.sleep", return_value=None):  # Patch sleep to speed up the test
        nodes[0].leader_append_entries()

    # Ensure that all followers received the append_entries call
    for node in nodes[1:]:
        assert node.state == "follower"


def test_failed_election_due_to_low_votes(setup_nodes):
    nodes = setup_nodes

    # Simulate a situation where node 0 doesn't get enough votes
    def mock_request_vote(
        term, candidate_id, candidate_last_log_index, candidate_last_log_term
    ):
        return False if candidate_id == 0 else True

    with patch.object(RaftNode, "request_vote", side_effect=mock_request_vote):
        nodes[0].start_election()

    # Node 0 should fail to become the leader
    assert nodes[0].state == "follower"
    # Ensure node 0 did not receive the majority of votes
    assert nodes[0].vote_count == 1


def test_log_consistency_during_election(setup_nodes):
    nodes = setup_nodes

    # Simulate a situation where nodes have different log terms
    nodes[0].log = [{"term": 1}]
    nodes[1].log = [{"term": 2}]
    nodes[2].log = [{"term": 1}]
    nodes[3].log = [{"term": 1}]
    nodes[4].log = [{"term": 1}]

    # Node 1 has the most up-to-date log and should win the election
    with patch("time.sleep", return_value=None):  # Patch sleep to speed up the test
        nodes[1].start_election()

    assert nodes[1].state == "leader"
    assert all(node.state == "follower" for node in nodes if node != nodes[1])
