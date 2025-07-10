import jax
import jaxkd as jk

kp, kq = jax.random.split(jax.random.key(83))
points = jax.random.normal(kp, shape=(100_000, 3))
queries = jax.random.normal(kq, shape=(10_000, 3))

tree = jk.build_tree(points)
counts = jk.count_neighbors(tree, queries, r=0.1)
neighbors, distances = jk.query_neighbors(tree, queries, k=10)
