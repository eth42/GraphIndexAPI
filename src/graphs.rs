use std::collections::{HashMap, HashSet};

use crate::{bit_vectors::{BitVector, BitVectorMut}, heaps::MinHeap, types::{Float, UnsignedInteger}};


pub trait Graph<R: UnsignedInteger> {
	fn reserve(&mut self, n_vertices: usize);
	fn n_vertices(&self) -> usize;
	fn n_edges(&self) -> usize;
	#[inline(always)]
	fn degree(&self, vertex: R) -> usize { self.neighbors(vertex).len() }
	#[inline(always)]
	fn clear_neighbors(&mut self, vertex: R) {
		(0..self.neighbors(vertex).len()).rev().for_each(|i| self.remove_edge_by_index(vertex, i));
	}
	fn neighbors(&self, vertex: R) -> Vec<R>;
	#[inline(always)]
	fn foreach_neighbor<Fun: FnMut(&R)>(&self, vertex: R, f: Fun) {
		self.neighbors(vertex).iter().for_each(f);
	}
	fn foreach_neighbor_mut<Fun: FnMut(&mut R)>(&mut self, vertex: R, f: Fun);
	fn iter_neighbors<'a>(&'a self, vertex: R) -> impl Iterator<Item=&'a R>;
	fn add_node(&mut self);
	fn add_node_with_capacity(&mut self, capacity: usize);
	fn add_edge(&mut self, vertex1: R, vertex2: R);
	#[inline(always)]
	fn find_edge(&self, vertex1: R, vertex2: R) -> Option<usize> {
		self.neighbors(vertex1).iter().position(|&v| v == vertex2)
	}
	#[inline(always)]
	fn remove_edge(&mut self, vertex1: R, vertex2: R) {
		let index = self.find_edge(vertex1, vertex2);
		if index.is_some() {
			self.remove_edge_by_index(vertex1, unsafe{index.unwrap_unchecked()});
		}
	}
	fn remove_edge_by_index(&mut self, vertex: R, index: usize);
	#[inline(always)]
	fn remove_edges_chunk(&mut self, vertex1: R, vertices2: &Vec<R>) {
		let mut remove_indices = vertices2.iter()
		.map(|&v| self.find_edge(vertex1, v))
		.filter(|v| v.is_some())
		.map(|v| unsafe{v.unwrap_unchecked()})
		.collect::<Vec<_>>();
		remove_indices.sort();
		remove_indices.into_iter().rev().for_each(|i| self.remove_edge_by_index(vertex1, i));
	}
	#[inline(always)]
	fn add_edges_chunk(&mut self, vertex1: R, vertices2: &Vec<R>) {
		vertices2.iter().for_each(|&v| self.add_edge(vertex1, v));
	}
	fn ego_graph_nodes_hashset(&self, vertex: R, radius: usize) -> Vec<R> {
		let mut visited = HashSet::new();
		let mut work = vec![vertex];
		let mut visited_ids = vec![vertex];
		visited.insert(vertex);
		for _ in 0..radius {
			let mut next_work = Vec::new();
			for &v in &work {
				for &n in &self.neighbors(v) {
					if !visited.contains(&n) {
						visited.insert(n);
						next_work.push(n);
						visited_ids.push(n);
					}
				}
			}
			work = next_work;
		}
		visited_ids
	}
	fn ego_graph_nodes_bitvec(&self, vertex: R, radius: usize) -> Vec<R> {
		let mut visited = vec![0u64; (self.n_vertices()+63)/64];
		let mut work = vec![vertex];
		let mut visited_ids = vec![vertex];
		visited.set_bit_unchecked(vertex.to_usize().unwrap(), true);
		for _ in 0..radius {
			let mut next_work = Vec::new();
			for &v in &work {
				for &n in &self.neighbors(v) {
					if !visited.get_bit_unchecked(n.to_usize().unwrap()) {
						visited.set_bit_unchecked(n.to_usize().unwrap(), true);
						next_work.push(n);
						visited_ids.push(n);
					}
				}
			}
			work = next_work;
		}
		visited_ids
	}
	fn as_dir_lol_graph(&self) -> DirLoLGraph<R> {
		let mut ret = DirLoLGraph::new();
		(0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).for_each(|i| ret.add_node_with_capacity(self.degree(i)));
		(0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).for_each(|i| {
			self.foreach_neighbor(i, |j| ret.add_edge(i, *j));
		});
		ret
	}
	fn as_undir_lol_graph(&self) -> UndirLoLGraph<R> {
		let mut ret: UndirLoLGraph<R> = UndirLoLGraph::new();
		(0..self.n_vertices()).for_each(|_| ret.add_node());
		if self.n_edges() == 0 { return ret; }
		/* Get edges in ascending node order */
		let mut edges = (0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).flat_map(|i| {
			self.neighbors(i).into_iter().map(move |j| (i.min(j),i.max(j)))
		}).collect::<Vec<_>>();
		/* Add edges if not equal to the previous one (unique edges) */
		edges.sort();
		ret.add_edge(edges[0].0, edges[0].1);
		(1..edges.len())
		.filter(|&i| edges[i] != edges[i-1])
		.for_each(|i| ret.add_edge(edges[i].0, edges[i].1));
		ret
	}
	fn as_fat_dir_graph(&self, id_remap: Option<Vec<R>>, n_vertices_override: Option<usize>, max_degree_override: Option<usize>) -> FatDirGraph<R> {
		if id_remap.is_some() { assert!(self.n_vertices() == id_remap.as_ref().unwrap().len()); }
		let n_vertices = n_vertices_override.unwrap_or(self.n_vertices());
		let max_degree = max_degree_override.unwrap_or((0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).map(|i| self.degree(i)).max().unwrap());
		let mut data = Vec::with_capacity(n_vertices*(max_degree+1));
		let mut n_edges = 0;
		unsafe{data.set_len(n_vertices*(max_degree+1));}
		(0..n_vertices).for_each(|i| data[i*(max_degree+1)] = R::zero());
		(0..self.n_vertices()).for_each(|i_usize| {
			let i = unsafe{R::from_usize(i_usize).unwrap_unchecked()};
			let degree = self.degree(i);
			n_edges += degree;
			let i_new = unsafe{id_remap.as_ref().map(|v| v[i_usize]).unwrap_or(i).to_usize().unwrap_unchecked()};
			let start = i_new * (max_degree+1);
			data[start] = unsafe{R::from(degree).unwrap_unchecked()};
			let start = start+1;
			self.iter_neighbors(i).enumerate().for_each(|(i_neighbor, j)| data[start+i_neighbor] = *j);
		});
		FatDirGraph {
			data,
			n_vertices,
			n_edges,
			max_degree,
		}
	}
	#[inline(always)]
	fn as_viewable_adj_graph(&self) -> Option<&impl ViewableAdjGraph<R>> { None::<&DirLoLGraph<R>> }
	#[inline(always)]
	fn as_viewable_adj_graph_mut(&mut self) -> Option<&mut impl ViewableAdjGraph<R>> { None::<&mut DirLoLGraph<R>> }
	#[inline(always)]
	fn as_vec_viewable_adj_graph(&self) -> Option<&impl VecViewableAdjGraph<R>> { None::<&DirLoLGraph<R>> }
	#[inline(always)]
	fn as_vec_viewable_adj_graph_mut(&mut self) -> Option<&mut impl VecViewableAdjGraph<R>> { None::<&mut DirLoLGraph<R>> }
}
pub trait WeightedGraph<R: UnsignedInteger, F: Float>: Graph<R> {
	fn edge_weight(&self, vertex1: R, vertex2: R) -> F;
	fn add_edge_with_weight(&mut self, vertex1: R, vertex2: R, weight: F);
	#[inline(always)]
	fn add_edges_with_weight_chunk(&mut self, vertex1: R, vertices2: &Vec<R>, weights: &Vec<F>) {
		vertices2.iter().zip(weights.iter())
		.for_each(|(&v,&w)| self.add_edge_with_weight(vertex1, v, w));
	}
	#[inline(always)]
	fn add_edges_with_zipped_weight_chunk(&mut self, vertex1: R, vertices2: &Vec<(F,R)>) {
		vertices2.iter()
		.for_each(|&(w,v)| self.add_edge_with_weight(vertex1, v, w));
	}
	fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>);
	fn neighbors_with_zipped_weights(&self, vertex: R) -> Vec<(F,R)>;
	#[inline(always)]
	fn foreach_neighbor_with_zipped_weight<Fun: FnMut(&F, &R)>(&self, vertex: R, mut f: Fun) {
		self.neighbors_with_zipped_weights(vertex).iter().for_each(|v| f(&v.0,&v.1));
	}
	fn foreach_neighbor_with_zipped_weight_mut<Fun: FnMut(&mut F, &mut R)>(&mut self, vertex: R, f: Fun);
	fn weighted_ego_graph_nodes(&self, vertex: R, radius: usize) -> (Vec<F>, Vec<R>) {
		let mut visited = HashMap::new();
		let mut work = vec![vertex];
		visited.insert(vertex, F::zero());
		for _ in 0..radius {
			let mut next_work = Vec::new();
			for &v in &work {
				let v_dist = visited[&v];
				let (neighbors, weights) = self.neighbors_with_weights(v);
				for (&w, &n) in neighbors.iter().zip(weights.iter()) {
					if !visited.contains_key(&n) || visited[&n] > v_dist + w {
						visited.insert(n, v_dist + w);
						next_work.push(n);
					}
				}
			}
			work = next_work;
		}
		let mut heap = MinHeap::new();
		for (&v, &d) in &visited {
			heap.push(d, v);
		}
		let mut visited_ids = Vec::with_capacity(heap.size());
		let mut visited_dists = Vec::with_capacity(heap.size());
		for (d, v) in heap.into_iter() {
			visited_ids.push(v);
			visited_dists.push(d);
		}
		(visited_dists, visited_ids)
	}
	fn as_weighted_dir_lol_graph(&self) -> WDirLoLGraph<R,F> {
		let mut ret = WDirLoLGraph::new();
		(0..self.n_vertices()).for_each(|_| ret.add_node());
		(0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).for_each(|i| {
			let (neighbors, weights) = self.neighbors_with_weights(i);
			neighbors.into_iter().zip(weights.into_iter()).for_each(|(d,j)| ret.add_edge_with_weight(i, j, d));
		});
		ret
	}
	fn as_weighted_undir_lol_graph(&self) -> WUndirLoLGraph<R,F> {
		let mut ret: WUndirLoLGraph<R,F> = WUndirLoLGraph::new();
		(0..self.n_vertices()).for_each(|_| ret.add_node());
		if self.n_edges() == 0 { return ret; }
		/* Get edges in ascending node order */
		let mut edges = (0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).flat_map(|i| {
			let (neighbors, weights) = self.neighbors_with_weights(i);
			neighbors.into_iter().zip(weights.into_iter()).map(move |(d,j)| (i.min(j),i.max(j),d))
		}).collect::<Vec<_>>();
		/* Add edges if not equal to the previous one (unique edges) */
		/* If both directions are available with different edge weights, the behavior is undefined */
		edges.sort_by_key(|&(i,j,_)| (i,j));
		ret.add_edge(edges[0].0, edges[0].1);
		(1..edges.len())
		.filter(|&i| edges[i].0 != edges[i-1].0 || edges[i].1 != edges[i-1].1)
		.for_each(|i| ret.add_edge_with_weight(edges[i].0, edges[i].1, edges[i].2));
		ret
	}
	#[inline(always)]
	fn as_viewable_weighted_adj_graph(&self) -> Option<&impl ViewableWeightedAdjGraph<R,F>> { None::<&WDirLoLGraph<R,F>> }
	#[inline(always)]
	fn as_vec_viewable_weighted_adj_graph(&self) -> Option<&impl VecViewableWeightedAdjGraph<R,F>> { None::<&WDirLoLGraph<R,F>> }
}
pub trait ViewableAdjGraph<R: UnsignedInteger>: Graph<R> {
	fn view_neighbors(&self, vertex: R) -> &[R];
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [R];
}
pub trait VecViewableAdjGraph<R: UnsignedInteger>: Graph<R> {
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<R>;
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<R>;
}
pub trait ViewableWeightedAdjGraph<R: UnsignedInteger, F: Float>: WeightedGraph<R,F> {
	fn view_neighbors(&self, vertex: R) -> &[(F,R)];
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F,R)];
}
pub trait VecViewableWeightedAdjGraph<R: UnsignedInteger, F: Float>: WeightedGraph<R,F> {
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<(F,R)>;
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<(F,R)>;
}


pub struct DirLoLGraph<R: UnsignedInteger> {
	pub adjacency: Vec<Vec<R>>,
	pub n_edges: usize
}
impl<R: UnsignedInteger> DirLoLGraph<R> {
	#[inline(always)]
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
}
impl<R: UnsignedInteger> Graph<R> for DirLoLGraph<R> {
	#[inline(always)]
	fn reserve(&mut self, n_vertices: usize) {
		self.adjacency.reserve(n_vertices);
	}
	#[inline(always)]
	fn n_vertices(&self) -> usize {
		self.adjacency.len()
	}
	#[inline(always)]
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	#[inline(always)]
	fn degree(&self, vertex: R) -> usize {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).len()
		}
	}
	#[inline(always)]
	fn clear_neighbors(&mut self, vertex: R) {
		unsafe {
			let adj = self.adjacency.get_unchecked_mut(vertex.to_usize().unwrap_unchecked());
			self.n_edges -= adj.len();
			adj.clear();
		}
	}
	#[inline(always)]
	fn neighbors(&self, vertex: R) -> Vec<R> {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).clone()
		}
	}
	#[inline(always)]
	fn foreach_neighbor<Fun: FnMut(&R)>(&self, vertex: R, f: Fun) {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).iter().for_each(f);
		}
	}
	#[inline(always)]
	fn foreach_neighbor_mut<Fun: FnMut(&mut R)>(&mut self, vertex: R, f: Fun) {
		unsafe {
			self.adjacency.get_unchecked_mut(vertex.to_usize().unwrap_unchecked()).iter_mut().for_each(f);
		}
	}
	#[inline(always)]
	fn iter_neighbors<'a>(&'a self, vertex: R) -> impl Iterator<Item=&'a R> {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).iter()
		}
	}
	#[inline(always)]
	fn add_node(&mut self) {
		self.adjacency.push(Vec::new());
	}
	#[inline(always)]
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(Vec::with_capacity(capacity));
	}
	#[inline(always)]
	fn add_edge(&mut self, vertex1: R, vertex2: R) {
		unsafe {
			self.adjacency.get_unchecked_mut(vertex1.to_usize().unwrap_unchecked()).push(vertex2);
		}
		self.n_edges += 1;
	}
	#[inline(always)]
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		unsafe {
			self.adjacency.get_unchecked_mut(vertex.to_usize().unwrap_unchecked()).swap_remove(index);
		}
		self.n_edges -= 1;
	}
	#[inline(always)]
	fn as_viewable_adj_graph(&self) -> Option<&impl ViewableAdjGraph<R>> {
		Some(self)
	}
	#[inline(always)]
	fn as_viewable_adj_graph_mut(&mut self) -> Option<&mut impl ViewableAdjGraph<R>> {
		Some(self)
	}
	#[inline(always)]
	fn as_vec_viewable_adj_graph(&self) -> Option<&impl VecViewableAdjGraph<R>> {
		Some(self)
	}
	#[inline(always)]
	fn as_vec_viewable_adj_graph_mut(&mut self) -> Option<&mut impl VecViewableAdjGraph<R>> {
		Some(self)
	}
}
impl<R: UnsignedInteger> ViewableAdjGraph<R> for DirLoLGraph<R> {
	#[inline(always)]
	fn view_neighbors(&self, vertex: R) -> &[R] {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked())
		}
	}
	#[inline(always)]
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [R] {
		unsafe {
			self.adjacency.get_unchecked_mut(vertex.to_usize().unwrap_unchecked())
		}
	}
}
impl<R: UnsignedInteger> VecViewableAdjGraph<R> for DirLoLGraph<R> {
	#[inline(always)]
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<R> {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked())
		}
	}
	#[inline(always)]
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<R> {
		unsafe {
			self.adjacency.get_unchecked_mut(vertex.to_usize().unwrap_unchecked())
		}
	}
}
pub struct UndirLoLGraph<R: UnsignedInteger> {
	pub adjacency: Vec<Vec<R>>,
	pub n_edges: usize
}
impl<R: UnsignedInteger> UndirLoLGraph<R> {
	#[inline(always)]
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
}
impl<R: UnsignedInteger> Graph<R> for UndirLoLGraph<R> {
	#[inline(always)]
	fn reserve(&mut self, n_vertices: usize) {
		self.adjacency.reserve(n_vertices);
	}
	#[inline(always)]
	fn n_vertices(&self) -> usize {
		self.adjacency.len()
	}
	#[inline(always)]
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	#[inline(always)]
	fn clear_neighbors(&mut self, vertex: R) {
		unsafe {
			let adj = self.adjacency.get_unchecked_mut(vertex.to_usize().unwrap_unchecked());
			self.n_edges -= adj.len();
			adj.clear();
		}
	}
	#[inline(always)]
	fn neighbors(&self, vertex: R) -> Vec<R> {
		self.adjacency[vertex.to_usize().unwrap()].clone()
	}
	#[inline(always)]
	fn degree(&self, vertex: R) -> usize {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).len()
		}
	}
	#[inline(always)]
	fn foreach_neighbor<Fun: FnMut(&R)>(&self, vertex: R, f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter().for_each(f);
	}
	#[inline(always)]
	fn foreach_neighbor_mut<Fun: FnMut(&mut R)>(&mut self, vertex: R, f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter_mut().for_each(f);
	}
	#[inline(always)]
	fn iter_neighbors<'a>(&'a self, vertex: R) -> impl Iterator<Item=&'a R> {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).iter()
		}
	}
	#[inline(always)]
	fn add_node(&mut self) {
		self.adjacency.push(Vec::new());
	}
	#[inline(always)]
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(Vec::with_capacity(capacity));
	}
	#[inline(always)]
	fn add_edge(&mut self, vertex1: R, vertex2: R) {
		self.adjacency[vertex1.to_usize().unwrap()].push(vertex2);
		self.adjacency[vertex2.to_usize().unwrap()].push(vertex1);
		self.n_edges += 1;
	}
	#[inline(always)]
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		self.adjacency[vertex.to_usize().unwrap()].swap_remove(index);
		let neighbor = self.adjacency[vertex.to_usize().unwrap()][index];
		let neighbor_index = self.adjacency[neighbor.to_usize().unwrap()].iter().position(|&v| v == vertex).unwrap();
		self.adjacency[neighbor.to_usize().unwrap()].swap_remove(neighbor_index);
		self.n_edges -= 1;
	}
	#[inline(always)]
	fn as_viewable_adj_graph(&self) -> Option<&impl ViewableAdjGraph<R>> {
		Some(self)
	}
	#[inline(always)]
	fn as_viewable_adj_graph_mut(&mut self) -> Option<&mut impl ViewableAdjGraph<R>> {
		Some(self)
	}
	#[inline(always)]
	fn as_vec_viewable_adj_graph(&self) -> Option<&impl VecViewableAdjGraph<R>> {
		Some(self)
	}
	#[inline(always)]
	fn as_vec_viewable_adj_graph_mut(&mut self) -> Option<&mut impl VecViewableAdjGraph<R>> {
		Some(self)
	}
}
impl<R: UnsignedInteger> ViewableAdjGraph<R> for UndirLoLGraph<R> {
	#[inline(always)]
	fn view_neighbors(&self, vertex: R) -> &[R] {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	#[inline(always)]
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [R] {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
impl<R: UnsignedInteger> VecViewableAdjGraph<R> for UndirLoLGraph<R> {
	#[inline(always)]
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<R> {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	#[inline(always)]
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<R> {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
pub struct WDirLoLGraph<R: UnsignedInteger, F: Float> {
	pub adjacency: Vec<Vec<(F,R)>>,
	pub n_edges: usize
}
impl<R: UnsignedInteger, F: Float> WDirLoLGraph<R,F> {
	#[inline(always)]
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
}
impl<R: UnsignedInteger, F: Float> Graph<R> for WDirLoLGraph<R,F> {
	#[inline(always)]
	fn reserve(&mut self, n_vertices: usize) {
		self.adjacency.reserve(n_vertices);
	}
	#[inline(always)]
	fn n_vertices(&self) -> usize {
		self.adjacency.len()
	}
	#[inline(always)]
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	#[inline(always)]
	fn degree(&self, vertex: R) -> usize {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).len()
		}
	}
	#[inline(always)]
	fn clear_neighbors(&mut self, vertex: R) {
		unsafe {
			let adj = self.adjacency.get_unchecked_mut(vertex.to_usize().unwrap_unchecked());
			self.n_edges -= adj.len();
			adj.clear();
		}
	}
	#[inline(always)]
	fn neighbors(&self, vertex: R) -> Vec<R> {
		self.adjacency[vertex.to_usize().unwrap()].iter().map(|&(_,v)| v).collect()
	}
	#[inline(always)]
	fn foreach_neighbor<Fun: FnMut(&R)>(&self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter().for_each(|v|f(&v.1));
	}
	#[inline(always)]
	fn foreach_neighbor_mut<Fun: FnMut(&mut R)>(&mut self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter_mut().for_each(|v|f(&mut v.1));
	}
	#[inline(always)]
	fn iter_neighbors<'a>(&'a self, vertex: R) -> impl Iterator<Item=&'a R> {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).iter().map(|v|&v.1)
		}
	}
	#[inline(always)]
	fn add_node(&mut self) {
		self.adjacency.push(Vec::new());
	}
	#[inline(always)]
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(Vec::with_capacity(capacity));
	}
	#[inline(always)]
	fn add_edge(&mut self, _vertex1: R, _vertex2: R) {
		panic!("Cannot add edge without weight to a weighted graph");
	}
	#[inline(always)]
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		self.adjacency[vertex.to_usize().unwrap()].swap_remove(index);
		self.n_edges -= 1;
	}
}
impl<R: UnsignedInteger, F: Float> WeightedGraph<R,F> for WDirLoLGraph<R,F> {
	#[inline(always)]
	fn edge_weight(&self, vertex1: R, vertex2: R) -> F {
		self.adjacency[vertex1.to_usize().unwrap()].iter().find(|&&(_,v)| v == vertex2).unwrap().0
	}
	#[inline(always)]
	fn add_edge_with_weight(&mut self, vertex1: R, vertex2: R, weight: F) {
		self.adjacency[vertex1.to_usize().unwrap()].push((weight, vertex2));
		self.n_edges += 1;
	}
	#[inline(always)]
	fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>) {
		let mut neighbors = Vec::new();
		let mut weights = Vec::new();
		for &(w,v) in &self.adjacency[vertex.to_usize().unwrap()] {
			neighbors.push(v);
			weights.push(w);
		}
		(weights, neighbors)
	}
	#[inline(always)]
	fn neighbors_with_zipped_weights(&self, vertex: R) -> Vec<(F,R)> {
		self.adjacency[vertex.to_usize().unwrap()].clone()
	}
	#[inline(always)]
	fn foreach_neighbor_with_zipped_weight<Fun: FnMut(&F, &R)>(&self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter().for_each(|v| f(&v.0,&v.1));
	}
	#[inline(always)]
	fn foreach_neighbor_with_zipped_weight_mut<Fun: FnMut(&mut F, &mut R)>(&mut self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter_mut().for_each(|(w,v)| f(w,v));
	}
	#[inline(always)]
	fn as_viewable_weighted_adj_graph(&self) -> Option<&impl ViewableWeightedAdjGraph<R,F>> {
		Some(self)
	}
	#[inline(always)]
	fn as_vec_viewable_weighted_adj_graph(&self) -> Option<&impl VecViewableWeightedAdjGraph<R,F>> {
		Some(self)
	}
}
impl<R: UnsignedInteger, F: Float> ViewableWeightedAdjGraph<R,F> for WDirLoLGraph<R,F> {
	#[inline(always)]
	fn view_neighbors(&self, vertex: R) -> &[(F,R)] {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	#[inline(always)]
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F,R)] {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
impl<R: UnsignedInteger, F: Float> VecViewableWeightedAdjGraph<R,F> for WDirLoLGraph<R,F> {
	#[inline(always)]
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<(F,R)> {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	#[inline(always)]
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<(F,R)> {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
pub struct WUndirLoLGraph<R: UnsignedInteger, F: Float> {
	pub adjacency: Vec<Vec<(F,R)>>,
	pub n_edges: usize
}
impl<R: UnsignedInteger, F: Float> WUndirLoLGraph<R,F> {
	#[inline(always)]
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
}
impl<R: UnsignedInteger, F: Float> Graph<R> for WUndirLoLGraph<R,F> {
	#[inline(always)]
	fn reserve(&mut self, n_vertices: usize) {
		self.adjacency.reserve(n_vertices);
	}
	#[inline(always)]
	fn n_vertices(&self) -> usize {
		self.adjacency.len()
	}
	#[inline(always)]
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	#[inline(always)]
	fn degree(&self, vertex: R) -> usize {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).len()
		}
	}
	#[inline(always)]
	fn neighbors(&self, vertex: R) -> Vec<R> {
		self.adjacency[vertex.to_usize().unwrap()].iter().map(|&(_,v)| v).collect()
	}
	#[inline(always)]
	fn clear_neighbors(&mut self, vertex: R) {
		unsafe {
			let adj = self.adjacency.get_unchecked_mut(vertex.to_usize().unwrap_unchecked());
			self.n_edges -= adj.len();
			adj.clear();
		}
	}
	#[inline(always)]
	fn foreach_neighbor<Fun: FnMut(&R)>(&self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter().for_each(|v|f(&v.1));
	}
	#[inline(always)]
	fn foreach_neighbor_mut<Fun: FnMut(&mut R)>(&mut self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter_mut().for_each(|v|f(&mut v.1));
	}
	#[inline(always)]
	fn iter_neighbors<'a>(&'a self, vertex: R) -> impl Iterator<Item=&'a R> {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).iter().map(|v|&v.1)
		}
	}
	#[inline(always)]
	fn add_node(&mut self) {
		self.adjacency.push(Vec::new());
	}
	#[inline(always)]
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(Vec::with_capacity(capacity));
	}
	#[inline(always)]
	fn add_edge(&mut self, _vertex1: R, _vertex2: R) {
		panic!("Cannot add edge without weight to a weighted graph");
	}
	#[inline(always)]
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		self.adjacency[vertex.to_usize().unwrap()].swap_remove(index);
		let neighbor = self.adjacency[vertex.to_usize().unwrap()][index].0;
		let neighbor_index = self.adjacency[neighbor.to_usize().unwrap()].iter().position(|&v| v.1 == vertex).unwrap();
		self.adjacency[neighbor.to_usize().unwrap()].swap_remove(neighbor_index);
		self.n_edges -= 1;
	}
}
impl<R: UnsignedInteger, F: Float> WeightedGraph<R,F> for WUndirLoLGraph<R,F> {
	#[inline(always)]
	fn edge_weight(&self, vertex1: R, vertex2: R) -> F {
		self.adjacency[vertex1.to_usize().unwrap()].iter().find(|&&(_,v)| v == vertex2).unwrap().0
	}
	#[inline(always)]
	fn add_edge_with_weight(&mut self, vertex1: R, vertex2: R, weight: F) {
		self.adjacency[vertex1.to_usize().unwrap()].push((weight, vertex2));
		self.adjacency[vertex2.to_usize().unwrap()].push((weight, vertex1));
		self.n_edges += 1;
	}
	#[inline(always)]
	fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>) {
		let mut neighbors = Vec::new();
		let mut weights = Vec::new();
		for &(w,v) in &self.adjacency[vertex.to_usize().unwrap()] {
			neighbors.push(v);
			weights.push(w);
		}
		(weights, neighbors)
	}
	#[inline(always)]
	fn neighbors_with_zipped_weights(&self, vertex: R) -> Vec<(F,R)> {
		self.adjacency[vertex.to_usize().unwrap()].clone()
	}
	#[inline(always)]
	fn foreach_neighbor_with_zipped_weight<Fun: FnMut(&F, &R)>(&self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter().for_each(|v| f(&v.0,&v.1));
	}
	#[inline(always)]
	fn foreach_neighbor_with_zipped_weight_mut<Fun: FnMut(&mut F, &mut R)>(&mut self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter_mut().for_each(|(w,v)| f(w,v));
	}
	#[inline(always)]
	fn as_viewable_weighted_adj_graph(&self) -> Option<&impl ViewableWeightedAdjGraph<R,F>> {
		Some(self)
	}
	#[inline(always)]
	fn as_vec_viewable_weighted_adj_graph(&self) -> Option<&impl VecViewableWeightedAdjGraph<R,F>> {
		Some(self)
	}
}
impl<R: UnsignedInteger, F: Float> ViewableWeightedAdjGraph<R,F> for WUndirLoLGraph<R,F> {
	#[inline(always)]
	fn view_neighbors(&self, vertex: R) -> &[(F,R)] {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	#[inline(always)]
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F,R)] {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
impl<R: UnsignedInteger, F: Float> VecViewableWeightedAdjGraph<R,F> for WUndirLoLGraph<R,F> {
	#[inline(always)]
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<(F,R)> {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	#[inline(always)]
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<(F,R)> {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}



pub struct FatDirGraph<R: UnsignedInteger> {
	pub data: Vec<R>,
	pub n_vertices: usize,
	pub max_degree: usize,
	pub n_edges: usize,
}
impl<R: UnsignedInteger> FatDirGraph<R> {
	#[inline(always)]
	pub fn new(max_degree: usize) -> Self {
		Self{data: Vec::new(), n_vertices: 0, max_degree: max_degree, n_edges: 0}
	}
}
impl<R: UnsignedInteger> Graph<R> for FatDirGraph<R> {
	#[inline(always)]
	fn reserve(&mut self, n_vertices: usize) {
		self.data.reserve((self.max_degree+1)*n_vertices);
	}
	#[inline(always)]
	fn n_vertices(&self) -> usize {
		self.n_vertices
	}
	#[inline(always)]
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	#[inline(always)]
	fn degree(&self, vertex: R) -> usize {
		unsafe {
			let start = vertex.to_usize().unwrap_unchecked() * (self.max_degree+1);
			self.data.get_unchecked(start).to_usize().unwrap_unchecked()
		}
	}
	#[inline(always)]
	fn clear_neighbors(&mut self, vertex: R) {
		unsafe {
			let start = vertex.to_usize().unwrap_unchecked() * (self.max_degree+1);
			*self.data.get_unchecked_mut(start) = R::zero()
		}
	}
	#[inline(always)]
	fn neighbors(&self, vertex: R) -> Vec<R> {
		self.view_neighbors(vertex).iter().cloned().collect()
	}
	#[inline(always)]
	fn foreach_neighbor<Fun: FnMut(&R)>(&self, vertex: R, f: Fun) {
		self.view_neighbors(vertex).iter().for_each(f);
	}
	#[inline(always)]
	fn foreach_neighbor_mut<Fun: FnMut(&mut R)>(&mut self, vertex: R, f: Fun) {
		self.view_neighbors_mut(vertex).iter_mut().for_each(f);
	}
	#[inline(always)]
	fn iter_neighbors<'a>(&'a self, vertex: R) -> impl Iterator<Item=&'a R> {
		self.view_neighbors(vertex).iter()
	}
	#[inline(always)]
	fn add_node(&mut self) {
		self.data.reserve(self.max_degree+1);
		let curr_len = self.data.len();
		unsafe{self.data.set_len(curr_len+self.max_degree+1);}
		self.data[curr_len] = R::zero();
		self.n_vertices += 1;
	}
	#[inline(always)]
	fn add_node_with_capacity(&mut self, capacity: usize) {
		assert!(capacity <= self.max_degree);
		self.add_node();
	}
	#[inline(always)]
	fn add_edge(&mut self, vertex1: R, vertex2: R) {
		unsafe {
			let start = vertex1.to_usize().unwrap_unchecked() * (self.max_degree+1);
			let n_neighbors = self.data.get_unchecked(start).to_usize().unwrap_unchecked();
			assert!(n_neighbors < self.max_degree);
			let end = start+1+n_neighbors;
			*self.data.get_unchecked_mut(end) = vertex2;
			*self.data.get_unchecked_mut(start) += R::one();
		}
		self.n_edges += 1;
	}
	#[inline(always)]
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		unsafe {
			let start = vertex.to_usize().unwrap_unchecked() * (self.max_degree+1);
			let n_neighbors = self.data.get_unchecked(start).to_usize().unwrap_unchecked();
			assert!(n_neighbors > 0);
			let start = start+1;
			let end = start+n_neighbors;
			self.data.swap(start+index, end-1);
			*self.data.get_unchecked_mut(start-1) -= R::one();
		}
		self.n_edges -= 1;
	}
	#[inline(always)]
	fn as_viewable_adj_graph(&self) -> Option<&impl ViewableAdjGraph<R>> {
		Some(self)
	}
	#[inline(always)]
	fn as_viewable_adj_graph_mut(&mut self) -> Option<&mut impl ViewableAdjGraph<R>> {
		Some(self)
	}
}
impl<R: UnsignedInteger> ViewableAdjGraph<R> for FatDirGraph<R> {
	#[inline(always)]
	fn view_neighbors(&self, vertex: R) -> &[R] {
		unsafe {
			let start = vertex.to_usize().unwrap_unchecked() * (self.max_degree+1);
			let n_neighbors = self.data.get_unchecked(start).to_usize().unwrap_unchecked();
			let start = start+1;
			let end = start+n_neighbors;
			self.data.get_unchecked(start..end)
		}
	}
	#[inline(always)]
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [R] {
		unsafe {
			let start = vertex.to_usize().unwrap_unchecked() * (self.max_degree+1);
			let n_neighbors = self.data.get_unchecked(start).to_usize().unwrap_unchecked();
			let start = start+1;
			let end = start+n_neighbors;
			self.data.get_unchecked_mut(start..end)
		}
	}
}
