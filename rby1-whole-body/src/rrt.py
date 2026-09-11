import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import scipy
import time

class RRTOptions:
	def __init__(self, step_size=1e-1, check_size=1e-2, max_vertices=1e3,
	             max_iters=1e4, goal_sample_frequency=0.05, always_swap=False,
	             timeout=np.inf):
		self.step_size = step_size
		self.check_size = check_size
		self.max_vertices = int(max_vertices)
		self.max_iters = int(max_iters)
		self.goal_sample_frequency = goal_sample_frequency
		self.always_swap = always_swap
		self.timeout = timeout
		assert self.goal_sample_frequency >= 0
		assert self.goal_sample_frequency <= 1

def _tree_coords(tree):
	"""Configurations of ``tree``'s nodes as a contiguous ``(n, d)`` array.

	Maintained incrementally in ``tree.graph`` and grown by doubling, because the
	nearest-neighbour query is this planner's inner loop and it was the binding cost.
	The previous implementation built a Python list of ``Distance()`` calls on every
	query, so the search was O(n^2) with a Python-level constant per node.

	Measured on the constrained lift leg: that held the tree to 1544-3328 vertices
	inside the 120 s timeout against a ``max_vertices`` cap of 60,000 -- the caps were
	unreachable by ~20x, wall-clock was what actually stopped the search, and raising
	the caps therefore bought nothing. Vectorising this is what makes the caps mean
	something.

	Nodes are only ever appended under indices ``0..len(tree)-1`` (see the
	``add_node(len(tree), ...)`` calls), so syncing is just copying in whatever
	arrived since the last query.
	"""
	buf = tree.graph.get("_coord_buf")
	n_cached = tree.graph.get("_coord_n", 0)
	n = len(tree)
	if buf is None:
		buf = np.empty((max(1024, 2 * n), len(tree.nodes[0]["q"])), dtype=float)
		n_cached = 0
	if n > buf.shape[0]:
		grown = np.empty((max(2 * n, 2 * buf.shape[0]), buf.shape[1]), dtype=float)
		grown[:n_cached] = buf[:n_cached]
		buf = grown
	for i in range(n_cached, n):
		buf[i] = tree.nodes[i]["q"]
	tree.graph["_coord_buf"] = buf
	tree.graph["_coord_n"] = n
	return buf[:n]


def _extreme_idx(tree, q, default_distance, Distance, furthest=False):
	"""argmax/argmin of distance from ``q`` over ``tree``'s nodes.

	Uses the vectorised path only for the default Euclidean metric, where squared
	distance gives an identical ordering. A caller-supplied ``Distance`` may not be a
	metric at all, so it keeps the original per-node loop.
	"""
	if default_distance:
		diff = _tree_coords(tree) - np.asarray(q, dtype=float)
		d2 = np.einsum("ij,ij->i", diff, diff)
		return int(np.argmax(d2) if furthest else np.argmin(d2))
	dists = [Distance(q, tree.nodes[i]["q"]) for i in range(len(tree))]
	return int(np.argmax(dists) if furthest else np.argmin(dists))


class RRT:
	def __init__(self, RandomConfig, ValidityChecker, Distance=None, EdgeValidator=None):
		self.RandomConfig = RandomConfig
		self.ValidityChecker = ValidityChecker
		self.EdgeValidator = EdgeValidator
		# Recorded so the nearest-neighbour query can take its vectorised path;
		# see _extreme_idx.
		self._default_distance = Distance is None
		if Distance is None:
			self.Distance = lambda x, y : np.linalg.norm(x - y)
		else:
			self.Distance = Distance

		self.options = None
		self.tree = None

	def plan(self, start, goal, options):
		t0 = time.time()
		self.options = options
		self.tree = nx.Graph()
		self.tree.add_node(0, q=start)
		success = False
		iters = tqdm(total=self.options.max_iters, position=0, desc="Iterations")
		vertices = tqdm(total=self.options.max_vertices, position=1, desc="Vertices")
		for i in range(self.options.max_iters):
			if time.time() - t0 > self.options.timeout:
				break
			iters.update(1)
			old_tree_size = len(self.tree)
			if len(self.tree) >= self.options.max_vertices or success == True:
				break
			sample_goal = np.random.random() < self.options.goal_sample_frequency
			q_subgoal = goal.copy() if sample_goal else self.RandomConfig()
			q_near_idx = self._nearest_idx(q_subgoal)
			while len(self.tree) < self.options.max_vertices:
				status = self._extend(q_near_idx, q_subgoal)
				q_new = self.tree.nodes[len(self.tree)-1]["q"]
				if self.Distance(q_new, goal) <= self.options.step_size:
					success = True
					break
				if status == "stopped" or status == "reached":
					break
				q_near_idx = len(self.tree)-1
			vertices.update(len(self.tree) - old_tree_size)

		if success:
			goal_idx = len(self.tree)
			self.tree.add_node(goal_idx, q=goal)
			self.tree.add_edge(goal_idx-1, goal_idx)
			start_idx = 0
			return self._path(start_idx, goal_idx)
		else:
			return []

	def _nearest_idx(self, q_subgoal):
		return _extreme_idx(self.tree, q_subgoal, self._default_distance,
					  self.Distance)

	def _furthest_idx(self, q_subgoal):
		return _extreme_idx(self.tree, q_subgoal, self._default_distance,
					  self.Distance, furthest=True)

	def _extend(self, q_near_idx, q_subgoal):
		q_near = self.tree.nodes[q_near_idx]["q"]
		step = q_subgoal - q_near
		unit_step = step / self.Distance(q_near, q_subgoal)
		q_new = q_near + self.options.step_size * unit_step
		validity_step = self.options.check_size * unit_step
		if self.ValidityChecker(q_new):
			if self.EdgeValidator and not self.EdgeValidator(q_near, q_new):
				return "stopped"
			prev_chk = q_near
			for i in range(1, int(self.options.step_size / self.options.check_size)):
				q_chk = q_near + i * validity_step
				if not self.ValidityChecker(q_chk):
					return "stopped"
				if self.EdgeValidator and not self.EdgeValidator(prev_chk, q_chk):
					return "stopped"
				prev_chk = q_chk
			q_new_idx = len(self.tree)
			self.tree.add_node(q_new_idx, q=q_new)
			self.tree.add_edge(q_near_idx, q_new_idx)
			dist_to_subgoal = self.Distance(q_new, q_subgoal)
			if dist_to_subgoal <= self.options.step_size:
				step = q_subgoal - q_new
				unit_step = step / dist_to_subgoal
				validity_step = self.options.check_size * unit_step
				prev_chk = q_new
				for i in range(1, int(dist_to_subgoal / self.options.check_size)):
					q_chk = q_new + i * validity_step
					if not self.ValidityChecker(q_chk):
						return "stopped"
					if self.EdgeValidator and not self.EdgeValidator(prev_chk, q_chk):
						return "stopped"
					prev_chk = q_chk
				if self.EdgeValidator and not self.EdgeValidator(prev_chk, q_subgoal):
					return "stopped"
				return "reached"
			else:
				return "extended"
		else:
			return "stopped"

	def _path(self, i, j):
		path_idx = nx.shortest_path(self.tree, source=i, target=j)
		path = [self.tree.nodes[idx]["q"] for idx in path_idx]
		return path

	def furthest_path(self):
		return self._path(0, self._furthest_idx(self.tree.nodes[0]["q"]))

	def nodes(self):
		return np.array([self.tree.nodes[i]["q"] for i in range(len(self.tree))])

	def adj_mat(self):
		return nx.adjacency_matrix(self.tree).toarray()

class BiRRT:
	def __init__(self, RandomConfig, ValidityChecker, Distance=None, EdgeValidator=None):
		self.RandomConfig = RandomConfig
		self.ValidityChecker = ValidityChecker
		self.EdgeValidator = EdgeValidator
		# Recorded so the nearest-neighbour query can take its vectorised path;
		# see _extreme_idx.
		self._default_distance = Distance is None
		if Distance is None:
			self.Distance = lambda x, y : np.linalg.norm(x - y)
		else:
			self.Distance = Distance

		self.options = None
		self.tree_a = None
		self.tree_b = None

	def plan(self, start, goal, options):
		t0 = time.time()

		self.options = options
		self.tree_a = nx.Graph()
		self.tree_a.add_node(0, q=start)
		self.tree_b = nx.Graph()
		self.tree_b.add_node(0, q=goal)

		success = False
		iters = tqdm(total=self.options.max_iters, position=0, desc="Iterations")
		vertices = tqdm(total=self.options.max_vertices, position=1, desc="Vertices")
		for i in range(self.options.max_iters):
			if time.time() - t0 > self.options.timeout:
				break
			iters.update(1)

			old_tree_size = len(self.tree_a) + len(self.tree_b)
			if old_tree_size >= self.options.max_vertices or success == True:
				break
			
			q_subgoal = self.RandomConfig()
			q_near_idx = self._nearest_idx(self.tree_a, q_subgoal)
			nodes_added = 0
			while len(self.tree_a) + len(self.tree_b) < self.options.max_vertices:
				status = self._extend(self.tree_a, q_near_idx, q_subgoal)
				if status == "stopped":
					break
				else:
					nodes_added += 1
				if status == "reached":
					break
				q_near_idx = len(self.tree_a)-1
			if nodes_added == 0:
				if self.options.always_swap:
					self.tree_a, self.tree_b = self.tree_b, self.tree_a
				continue
			vertices.update(len(self.tree_a) + len(self.tree_b) - old_tree_size)
			old_tree_size = len(self.tree_a) + len(self.tree_b)

			selected = np.random.randint(1, nodes_added+1)
			q_subgoal_idx = len(self.tree_a) - selected
			q_subgoal = self.tree_a.nodes[q_subgoal_idx]["q"]
			q_near_idx = self._nearest_idx(self.tree_b, q_subgoal)

			while len(self.tree_a) + len(self.tree_b) < self.options.max_vertices:
				status = self._extend(self.tree_b, q_near_idx, q_subgoal)
				if status == "stopped":
					break
				elif status == "reached":
					success = True
					break
				q_near_idx = len(self.tree_b)-1
			vertices.update(len(self.tree_a) + len(self.tree_b) - old_tree_size)
			self.tree_a, self.tree_b = self.tree_b, self.tree_a

		if success:
			path_a = self._path(self.tree_a, 0, len(self.tree_a)-1)
			path_b = self._path(self.tree_b, q_subgoal_idx, 0)
			path = path_a + path_b
			if np.linalg.norm(path[0] - start) > 1e-15:
				path.reverse()
			return path
		else:
			return []


	def _nearest_idx(self, tree, q_subgoal):
		return _extreme_idx(tree, q_subgoal, self._default_distance,
					  self.Distance)

	def _furthest_idx(self, tree, q_subgoal):
		return _extreme_idx(tree, q_subgoal, self._default_distance,
					  self.Distance, furthest=True)

	def _extend(self, tree, q_near_idx, q_subgoal):
		q_near = tree.nodes[q_near_idx]["q"]
		step = q_subgoal - q_near
		unit_step = step / self.Distance(q_near, q_subgoal)
		q_new = q_near + self.options.step_size * unit_step
		validity_step = self.options.check_size * unit_step
		if self.ValidityChecker(q_new):
			if self.EdgeValidator and not self.EdgeValidator(q_near, q_new):
				return "stopped"
			prev_chk = q_near
			for i in range(1, int(self.options.step_size / self.options.check_size)):
				q_chk = q_near + i * validity_step
				if not self.ValidityChecker(q_chk):
					return "stopped"
				if self.EdgeValidator and not self.EdgeValidator(prev_chk, q_chk):
					return "stopped"
				prev_chk = q_chk
			q_new_idx = len(tree)
			tree.add_node(q_new_idx, q=q_new)
			tree.add_edge(q_near_idx, q_new_idx)
			dist_to_subgoal = self.Distance(q_new, q_subgoal)
			if dist_to_subgoal <= self.options.step_size:
				step = q_subgoal - q_new
				unit_step = step / dist_to_subgoal
				validity_step = self.options.check_size * unit_step
				prev_chk = q_new
				for i in range(1, int(dist_to_subgoal / self.options.check_size)):
					q_chk = q_new + i * validity_step
					if not self.ValidityChecker(q_chk):
						return "stopped"
					if self.EdgeValidator and not self.EdgeValidator(prev_chk, q_chk):
						return "stopped"
					prev_chk = q_chk
				if self.EdgeValidator and not self.EdgeValidator(prev_chk, q_subgoal):
					return "stopped"
				return "reached"
			else:
				return "extended"
		else:
			return "stopped"

	def _path(self, tree, i, j):
		path_idx = nx.shortest_path(tree, source=i, target=j)
		path = [tree.nodes[idx]["q"] for idx in path_idx]
		return path

	def furthest_path(self):
		path_a = self._path(self.tree_a, 0,
			self._furthest_idx(self.tree_a, self.tree_a.nodes[0]["q"]))
		path_b = self._path(self.tree_b, 0,
			self._furthest_idx(self.tree_b, self.tree_b.nodes[0]["q"]))
		return path_a + path_b

	def nodes(self):
		nodes_a = np.array([self.tree_a.nodes[i]["q"] for i in range(len(self.tree_a))])
		nodes_b = np.array([self.tree_b.nodes[i]["q"] for i in range(len(self.tree_b))])
		return np.vstack((nodes_a, nodes_b))

	def adj_mat(self):
		adj_mat_a = nx.adjacency_matrix(self.tree_a).toarray()
		adj_mat_b = nx.adjacency_matrix(self.tree_b).toarray()
		full_adj_mat = scipy.linalg.block_diag(adj_mat_a, adj_mat_b)
		idx = len(self.tree_a) - 1
		full_adj_mat[idx,-1] = full_adj_mat[-1,idx] = 1
		return full_adj_mat