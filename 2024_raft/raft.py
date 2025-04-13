import random
import time
from typing import List, Optional


class RaftNode:
    def __init__(self, node_id: int, peers: List["RaftNode"]) -> None:
        """
        Initializes a Raft node.

        Args:
            node_id (int): The unique ID of the node.
            peers (List[RaftNode]): The list of peer nodes in the cluster.
        """
        self.node_id: int = node_id
        self.peers: List[RaftNode] = peers
        self.state: str = "follower"
        self.current_term: int = 0
        self.voted_for: Optional[int] = None
        self.log: List[dict] = []
        self.vote_count: int = 0

    def start_election(self) -> None:
        """
        Initiates an election process.
        - The node transitions to the candidate state.
        - The node increments its term and votes for itself.
        - The node sends vote requests to all peer nodes.
        - If the node receives a majority of votes, it becomes the leader.
        """
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.node_id
        self.vote_count = 1  # Vote for self
        print(f"Node {self.node_id} starts election for term {self.current_term}")

        # Request votes from peers
        for peer in self.peers:
            if peer.request_vote(
                self.current_term, self.node_id, len(self.log), self.get_last_log_term()
            ):
                self.vote_count += 1

        # Check if this node received the majority of votes
        if self.vote_count > len(self.peers) // 2:
            self.state = "leader"
            print(f"Node {self.node_id} becomes leader for term {self.current_term}")
            self.leader_append_entries()
        else:
            print(
                f"Node {self.node_id} failed to become leader. Total votes: {self.vote_count}"
            )
            self.state = "follower"

    def request_vote(
        self,
        term: int,
        candidate_id: int,
        candidate_last_log_index: int,
        candidate_last_log_term: int,
    ) -> bool:
        """
        Processes a vote request from a candidate node.

        Args:
            term (int): The term of the candidate.
            candidate_id (int): The ID of the candidate requesting the vote.
            candidate_last_log_index (int): The index of the candidate's last log entry.
            candidate_last_log_term (int): The term of the candidate's last log entry.

        Returns:
            bool: True if the vote is granted, False otherwise.
        """
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.state = "follower"

        # Grant vote if the candidate's log is at least as up-to-date as the voter's log
        if (
            self.voted_for is None or self.voted_for == candidate_id
        ) and self.is_candidate_log_up_to_date(
            candidate_last_log_index, candidate_last_log_term
        ):
            self.voted_for = candidate_id
            print(f"Node {self.node_id} votes for {candidate_id} in term {term}")
            return True
        return False

    def is_candidate_log_up_to_date(
        self, candidate_last_log_index: int, candidate_last_log_term: int
    ) -> bool:
        """
        Checks if the candidate's log is as up-to-date as this node's log.

        Args:
            candidate_last_log_index (int): The index of the candidate's last log entry.
            candidate_last_log_term (int): The term of the candidate's last log entry.

        Returns:
            bool: True if the candidate's log is up-to-date, False otherwise.
        """
        last_log_term = self.get_last_log_term()
        if candidate_last_log_term > last_log_term:
            return True
        if candidate_last_log_term == last_log_term:
            return candidate_last_log_index >= len(self.log)
        return False

    def get_last_log_term(self) -> int:
        """
        Retrieves the term of the last log entry.

        Returns:
            int: The term of the last log entry, or 0 if the log is empty.
        """
        if len(self.log) > 0:
            return self.log[-1]["term"]
        else:
            return 0

    def leader_append_entries(self) -> None:
        """
        Sends heartbeat (AppendEntries RPC) to all follower nodes periodically.
        - This simulates the leader sending heartbeats to maintain its authority.
        - The method runs a few iterations to simulate the behavior.
        """
        for _ in range(3):
            if self.state != "leader":
                break
            print(f"Leader {self.node_id} sending heartbeats to followers")
            for peer in self.peers:
                peer.append_entries(self.current_term, self.node_id, [])
            time.sleep(1)

    def append_entries(self, term: int, leader_id: int, entries: List[dict]) -> bool:
        """
        Handles the AppendEntries RPC (heartbeat or log replication) from the leader.

        Args:
            term (int): The term of the leader sending the entries.
            leader_id (int): The ID of the leader.
            entries (List[dict]): The log entries to be appended (empty for heartbeats).

        Returns:
            bool: True if the entries were accepted, False otherwise.
        """
        if term >= self.current_term:
            self.current_term = term
            self.state = "follower"
            print(f"Node {self.node_id} accepts entries from leader {leader_id}")
            return True
        return False


# Example usage:
if __name__ == "__main__":
    # Create a cluster of 5 nodes
    nodes = [RaftNode(i, []) for i in range(5)]
    for i in range(5):
        nodes[i].peers = nodes[:i] + nodes[i + 1 :]

    # Simulate the election process by starting an election on one of the nodes
    nodes[0].start_election()
