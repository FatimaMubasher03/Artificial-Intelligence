def dls(tree, cur, tar, d):
    if cur == tar:
        return True
    if d<= 0:
        return False
    for n in tree[cur]:
        if dls(tree, n, tar, d- 1):
            return True
    return False
def iddfs(tree, start, t, max):
    for depth in range(max):
        if dls(tree, start, t, depth):
            return True
    return False

tree = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start = 'A'
target = 'F'
max = 3

if iddfs(tree, start, target, max):
    print(f"Target {target} found within depth {max}")
else:
    print(f"Target {target} not found within depth {max}")