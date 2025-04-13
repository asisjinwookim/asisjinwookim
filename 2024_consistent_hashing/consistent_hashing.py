import hashlib
from typing import List, Tuple, Set


class ConsistentHashing:
    def __init__(self, vnodes_by_node: int = 3) -> None:
        self.ring: List[Tuple[int, str]] = []  # List of (hash_key, node) tuples
        self.vnodes_by_node: int = vnodes_by_node
        self.nodes: Set[str] = set()

    def _hash(self, key: str) -> int:
        """Generate a hash for the given key using MD5."""
        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)

    def _sort_ring(self) -> None:
        """Sort the ring based on hash keys."""
        self.ring.sort()

    def add_node(self, node: str) -> None:
        """Add a node and its virtual nodes to the ring."""
        self.nodes.add(node)
        for i in range(self.vnodes_by_node):
            vnode_key = f"{node}:{i}"
            hash_key = self._hash(vnode_key)
            self.ring.append((hash_key, node))
            print(f"add_node: virtual node {vnode_key}, ring length: {len(self.ring)}")
        self._sort_ring()
        self.redistribute_keys(node)

    def remove_node(self, node: str) -> None:
        """Remove a node and its virtual nodes from the ring."""
        self.nodes.remove(node)
        self.ring = [(hash_key, n) for hash_key, n in self.ring if n != node]
        print(f"remove_node: node {node}, ring length: {len(self.ring)}")
        self._sort_ring()
        self.redistribute_keys(node)

    def get_node(self, key: str) -> None | int | tuple[int, str]:
        """
        Get the node responsible for the given key.

        This method uses binary search to find the node responsible for the given key.

        The time complexity of this operation is as follows:
        - T(N) = O(log N): N is the number of nodes in the ring.

        Explanation:
        1. The method uses binary search to find the correct node for the given key.
        2. Therefore, the time complexity of this operation is O(log N).

        Also, when adding or deleting key-value data, Only node search costs and I/O as overhead, so the complexity is O(log N).

        """
        if not self.ring:
            return None
        hash_key = self._hash(key)
        print(f"get_node: key {key}, hash_key {hash_key}")
        # Perform binary search
        low, high = 0, len(self.ring) - 1
        while low <= high:
            mid = (low + high) // 2
            print(f"low {low}, high {high}, mid {mid}, ring[mid] {self.ring[mid]}")
            if self.ring[mid][0] >= hash_key:
                high = mid - 1
            else:
                low = mid + 1
        target_node = self.ring[low % len(self.ring)][1]
        print(f"get_node: target_node {target_node}")

        return target_node

    def redistribute_keys(self, node: str) -> None:
        """
        Redistribute keys when a node is added or removed.

        This method is called after adding or removing a node to redistribute keys.

        The time complexity of this operation is as follows:
        - T(N, M) = O(N/M * log N): N is the number of keys and M is the number of nodes.

        Explanation:
        1. Assuming that the number of keys is evenly distributed across nodes, the operation to move all keys of one
           node is O(N/M) on average, where N is the number of keys and M is the number of nodes.
        2. Finding the correct node for each key during redistribution takes O(log N) time because it uses a binary search.
           Therefore, the overall time complexity of this operation is O(N/M * log N).

        Args:
            node (str): The node to redistribute keys for.
        """
        keys_to_move = []
        for hash_key, n in self.ring:
            if n == node:
                keys_to_move.append(hash_key)
        for key_hash in keys_to_move:
            new_node = self.get_node(f"key:{key_hash}")
            print(f"Moving key {key_hash} to {new_node}")


# Example usage:
if __name__ == "__main__":
    ch = ConsistentHashing()

    # Add nodes
    ch.add_node("node1")
    ch.add_node("node2")

    # Add a new node and redistribute keys
    ch.add_node("node3")

    # Remove a node and redistribute keys
    ch.remove_node("node2")

    # Get node for a key
    node = ch.get_node("my_key")
    print(f"The node for 'my_key' is {node}")
