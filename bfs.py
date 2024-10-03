from collections import deque
def bfs(adj, s):
    q = deque()
    visited = [False] * len(adj)

    visited[s] = True
    q.append(s)
    while q:
        curr = q.popleft()
        print(curr, end=" ")
        for x in adj[curr]:
            if not visited[x]:
                visited[x] = True
                q.append(x)

def add_edge(adj, u, v):
    adj[u].append(v)
    adj[v].append(u)

if __name__ == "__main__":
    V = 5
    adj = [[] for _ in range(V)]
    add_edge(adj, 0, 1)
    add_edge(adj, 0, 2)
    add_edge(adj, 1, 3)
    add_edge(adj, 1, 4)
    add_edge(adj, 2, 4)
    print("BFS starting from 0: ")
    bfs(adj, 0)
