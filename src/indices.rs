use core::panic;

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use rayon::iter::IndexedParallelIterator;

use crate::data::MatrixDataSource;
use crate::graphs::{Graph, WDirLoLGraph};
use crate::heaps::{DualHeap, MaxHeap, MinHeap};
use crate::random::random_unique_uint;
use crate::types::{SyncUnsignedInteger, SyncFloat, trait_combiner};
use crate::measures::Distance;
use crate::sets::{HashOrBitset,HashSetLike};

pub trait IndexedDistance<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>>: MatrixDataSource<F> {
	fn distance<D1: Data<Elem=F>, D2: Data<Elem=F>>(&self, i: &ArrayBase<D1, Ix1>, j: &ArrayBase<D2, Ix1>) -> F;
	#[inline(always)]
	fn half_indexed_distance<D: Data<Elem=F>>(&self, i: R, q: &ArrayBase<D, Ix1>) -> F {
		unsafe { self.distance(&self.get_row(i.to_usize().unwrap_unchecked()), q) }
	}
	#[inline(always)]
	fn indexed_distance(&self, i: R, j: R) -> F {
		unsafe { self.distance(&self.get_row(i.to_usize().unwrap_unchecked()), &self.get_row(j.to_usize().unwrap_unchecked())) }
	}
}
pub trait RangeIndex<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>> {
	fn range_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix1>, range: F) -> (Array1<R>, Array1<F>);
	fn range_query_batch<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix2>, range: F) -> (Vec<Array1<R>>, Vec<Array1<F>>);
}
pub trait KnnIndex<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>> {
	fn knn_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix1>, k: usize) -> (Array1<R>, Array1<F>);
	fn knn_query_batch<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix2>, k: usize) -> (Array2<R>, Array2<F>);
}
trait_combiner!(GeneralIndex[R: SyncUnsignedInteger, F: SyncFloat, Dist: (Distance<F>)]: (RangeIndex<R, F, Dist>) + (KnnIndex<R, F, Dist>) + (IndexedDistance<R, F, Dist>) + (MatrixDataSource<F>));



pub fn bruteforce_neighbors<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	Dist: Distance<F>+Sync,
	DData: Data<Elem=F>+Sync,
	QData: Data<Elem=F>,
>(data: &ArrayBase<DData, Ix2>, queries: &ArrayBase<QData, Ix2>, dist: &Dist, k: usize) -> (Array2<R>, Array2<F>) {
	let nd = data.n_rows();
	let nq = queries.n_rows();
	/* Brute force queries */
	let mut bruteforce_ids: Array2<R> = Array2::zeros((nq, k));
	let mut bruteforce_dists: Array2<F> = Array2::zeros((nq, k));
	let n_threads = rayon::current_num_threads();
	let chunk_size = (nq + n_threads - 1) / n_threads;
	unsafe {
		bruteforce_ids.axis_chunks_iter_mut(Axis(0), chunk_size)
		.zip(bruteforce_dists.axis_chunks_iter_mut(Axis(0), chunk_size))
		.zip(queries.axis_chunks_iter(Axis(0), chunk_size))
		.map(|((a,b),c)|(a,b,c))
		.par_bridge()
		.for_each(|(mut id_chunk,mut dist_chunk,q_chunk)| {
			let mut dist_cache = Vec::with_capacity(nd);
			id_chunk.axis_iter_mut(Axis(0))
			.zip(dist_chunk.axis_iter_mut(Axis(0)))
			.zip(q_chunk.axis_iter(Axis(0)))
			.map(|((a,b),c)|(a,b,c))
			.for_each(|(mut ids_target, mut dists_target, q)| {
				dist_cache.clear();
				data.axis_iter(Axis(0)).enumerate()
				.for_each(|(i, x)| dist_cache.push((i,dist.dist(&q, &x))));
				dist_cache.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
				dist_cache[..k].iter().enumerate().for_each(|(i, &(idx, dist))| {
					ids_target[i] = R::from_usize(idx).unwrap_unchecked();
					dists_target[i] = dist;
				});
			});
		});
		(bruteforce_ids, bruteforce_dists)
	}
}


pub trait SearchCache<R: SyncUnsignedInteger, F: SyncFloat> {
	fn apply_local_id_map(&mut self, idx_map: &Vec<R>);
	fn extract_nn(&mut self, k_neighbors: usize) -> (Array1<R>, Array1<F>);
}
pub struct DefaultSearchCache<R: SyncUnsignedInteger, F: SyncFloat> {
	pub heap: MaxHeap<F,R>,
	pub visited_sets: Vec<HashOrBitset<R>>,
	// pub visited_set: crate::sets::BitSet<R>,
	pub frontier: MinHeap<F,R>,
}
impl<R: SyncUnsignedInteger, F: SyncFloat> DefaultSearchCache<R,F> {
	#[inline(always)]
	pub fn new(graph_sizes: Vec<usize>, max_heap_size: usize) -> Self {
		let visited_sets = graph_sizes.iter().map(|&size| HashOrBitset::new(size)).collect();
		// let mut visited_set = <crate::sets::BitSet<R> as HashSetLike<R>>::new(1_000_000);
		// visited_set.reserve(max_heap_size);
		Self{
			heap: MaxHeap::with_capacity(max_heap_size),
			visited_sets: visited_sets,
			frontier: MinHeap::with_capacity(max_heap_size),
		}
	}
	#[inline(always)]
	pub fn reserve(&mut self, max_heap_size: usize) {
		self.heap.reserve(max_heap_size);
		// self.visited_set.reserve(max_heap_size);
		self.frontier.reserve(max_heap_size);
	}
	#[inline(always)]
	pub fn clear(&mut self) {
		self.heap.clear();
		self.visited_sets.iter_mut().for_each(|set| set.clear());
		self.frontier.clear();
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat> SearchCache<R,F> for DefaultSearchCache<R,F> {
	#[inline(always)]
	fn apply_local_id_map(&mut self, idx_map: &Vec<R>) {
		self.heap.iter_mut().for_each(|(_, v)| *v = idx_map[unsafe{v.to_usize().unwrap_unchecked()}]);
	}
	fn extract_nn(&mut self, k_neighbors: usize) -> (Array1<R>, Array1<F>) {
		let mut ids = Array1::from_elem(k_neighbors, R::zero());
		let mut dists = Array1::from_elem(k_neighbors, F::zero());
		let n_in_heap = self.heap.size();
		let skipped = n_in_heap.max(k_neighbors) - k_neighbors;
		let max_index = n_in_heap.min(k_neighbors);
		self.heap.sorted_iter().skip(skipped).zip((0..max_index).rev())
		.for_each(|((d, v),i)| {
			ids[i] = v;
			dists[i] = d;
		});
		/* Return the result */
		(ids, dists)
	}
}
pub struct DefaultCappedSearchCache<R: SyncUnsignedInteger, F: SyncFloat> {
	pub heap: MaxHeap<F,R>,
	pub visited_sets: Vec<HashOrBitset<R>>,
	pub frontier: DualHeap<F,R>,
}
impl<R: SyncUnsignedInteger, F: SyncFloat> DefaultCappedSearchCache<R,F> {
	#[inline(always)]
	pub fn new(graph_sizes: Vec<usize>, max_heap_size: usize, max_frontier_size: usize) -> Self {
		let visited_sets = graph_sizes.iter().map(|&size| HashOrBitset::new(size)).collect();
		// visited_set.reserve(max_heap_size);
		Self{
			heap: MaxHeap::with_capacity(max_heap_size),
			visited_sets: visited_sets,
			frontier: DualHeap::with_capacity(max_frontier_size),
		}
	}
	#[inline(always)]
	pub fn reserve(&mut self, max_heap_size: usize, max_frontier_size: usize) {
		self.heap.reserve(max_heap_size);
		// self.visited_set.reserve(max_heap_size);
		self.frontier.reserve(max_frontier_size);
	}
	#[inline(always)]
	pub fn clear(&mut self) {
		self.heap.clear();
		self.visited_sets.iter_mut().for_each(|set| set.clear());
		self.frontier.clear();
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat> SearchCache<R,F> for DefaultCappedSearchCache<R,F> {
	#[inline(always)]
	fn apply_local_id_map(&mut self, idx_map: &Vec<R>) {
		self.heap.iter_mut().for_each(|(_, v)| *v = idx_map[unsafe{v.to_usize().unwrap_unchecked()}]);
	}
	fn extract_nn(&mut self, k_neighbors: usize) -> (Array1<R>, Array1<F>) {
		let mut ids = Array1::from_elem(k_neighbors, R::zero());
		let mut dists = Array1::from_elem(k_neighbors, F::zero());
		let n_in_heap = self.heap.size();
		let skipped = n_in_heap.max(k_neighbors) - k_neighbors;
		let max_index = n_in_heap.min(k_neighbors);
		self.heap.sorted_iter().skip(skipped).zip((0..max_index).rev())
		.for_each(|((d, v),i)| {
			ids[i] = v;
			dists[i] = d;
		});
		/* Return the result */
		(ids, dists)
	}
}


#[derive(Debug, Clone)]
pub struct NoSuchLayerError;
pub trait GraphIndex<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>>: GeneralIndex<R, F, Dist> + Sync {
	type SearchCache: SearchCache<R,F>;
	/// Create a simple cache
	fn _new_search_cache(&self, max_heap_size: usize) -> Self::SearchCache;
	/// Initializes a search cache for a new query.
	fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, entrypoints_override: Option<&Vec<R>>);
	/// Returns the number of layers in the graph index.
	fn layer_count(&self) -> usize;
	/// Returns the graph at the given layer if available, otherwise returns an error.
	/// 0 is the lowest level, layer_count()-1 is the highest level.
	fn get_layer(&self, layer: usize) -> Result<&impl Graph<R>, NoSuchLayerError>;
	/// Returns a map of the local graph node IDs to the global data IDs.
	/// If the layer does not have a mapping (i.e. local IDs are global IDs), None is returned.
	fn get_global_layer_ids(&self, layer: usize) -> Option<&Vec<R>>;
	/// Returns a map of the local graph node IDs to the local graph node IDs of the next layer.
	/// If the layer does not have a mapping (i.e. local IDs are identical), None is returned.
	fn get_local_layer_ids(&self, layer: usize) -> Option<&Vec<R>>;
	/// Executes a greedy search on the hierarchy with a maximum heap size and returns the heap containing the results.
	#[inline(always)]
	fn greedy_search<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, cache: &mut Self::SearchCache) -> (Array1<R>, Array1<F>) {
		self._init_cache(cache, q, k_neighbors, max_heap_size, None);
		/* Search all layers graph */
		self.greedy_search_with_cache(q, cache, max_heap_size);
		/* Extract the k nearest neighbors */
		cache.extract_nn(k_neighbors)
	}
	fn greedy_search_batch<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix2>, k_neighbors: usize, max_heap_size: usize) -> (Array2<R>, Array2<F>) {
		let mut ids = Array2::from_elem((q.dim().0, k_neighbors), R::zero());
		let mut dists = Array2::from_elem((q.dim().0, k_neighbors), F::zero());
		let n_threads = rayon::current_num_threads();
		let n_queries = q.dim().0;
		let batch_per_thread = (n_queries+n_threads-1)/n_threads;
		let raw_iter = ids.axis_chunks_iter_mut(Axis(0), batch_per_thread)
		.zip(dists.axis_chunks_iter_mut(Axis(0), batch_per_thread))
		.zip(q.axis_chunks_iter(Axis(0), batch_per_thread))
		.map(|((a,b),c)|(a,b,c)).collect::<Vec<_>>();
		raw_iter
		.into_par_iter()
		// .into_iter()
		.for_each(|(mut id_chunk, mut dist_chunk, chunk)| {
			let mut cache = self._new_search_cache(max_heap_size);
			id_chunk.axis_iter_mut(Axis(0))
			.zip(dist_chunk.axis_iter_mut(Axis(0)))
			.zip(chunk.axis_iter(Axis(0)))
			.map(|((ids, dists), q)| (ids, dists, q))
			.for_each(|(mut ids, mut dists, q)| {
				/* Fixme: This should ideally reuse the same heap memory for each search within a thread */
				let (ids_i, dists_i) = self.greedy_search(&q, k_neighbors, max_heap_size, &mut cache);
				ids.assign(&ids_i);
				dists.assign(&dists_i);
			});
		});
		(ids, dists)
	}
	/// Executes a greedy search on the given layer with a potentially pre-filled heap and returns the heap containing the results.
	fn greedy_search_layer_with_cache<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, cache: &mut Self::SearchCache, max_heap_size: usize, layer: usize);
	/// Executes a greedy search on the hierarchy with a potentially pre-filled heap and returns the heap containing the results.
	#[inline(always)]
	fn greedy_search_with_cache<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, cache: &mut Self::SearchCache, max_heap_size: usize) {
		for layer in (0..self.layer_count()).rev() {
			/* Get heap from the current layer */
			self.greedy_search_layer_with_cache(q, cache, max_heap_size, layer);
			/* Apply local ID map if available */
			let idx_map = self.get_local_layer_ids(layer);
			if idx_map.is_some() {
				cache.apply_local_id_map(unsafe{idx_map.unwrap_unchecked()});
			}
		}
	}
	/// Self join using the regular `greedy_search_batch` function
	fn self_join_query(&self, k_neighbors: usize, max_heap_size: usize) -> WDirLoLGraph<R,F> {
		let query = self.get_rows_slice(0, self.n_rows());
		let (ids, dists) = self.greedy_search_batch(&query, k_neighbors, max_heap_size);
		let adjacency = ids.axis_iter(Axis(0)).zip(dists.axis_iter(Axis(0)))
		.map(|(i_ids, i_dists)| {
			i_dists.into_iter().zip(i_ids.into_iter())
			.map(|(&w,&i)| (w,i)).collect()
		}).collect();
		WDirLoLGraph {
			adjacency: adjacency,
			n_edges: self.n_rows() * k_neighbors,
		}
	}
	/// Self join using the `greedy_search_layer_with_cache` function on the bottom layer
	fn self_join_query_local(&self, k_neighbors: usize, max_heap_size: usize) -> WDirLoLGraph<R,F> {
		let n_queries = self.n_rows();
		let query = self.get_rows_slice(0, n_queries);
		let n_threads = rayon::current_num_threads();
		let batch_per_thread = (n_queries+n_threads-1)/n_threads;
		let mut adjacency: Vec<Vec<(F,R)>> = (0..n_queries).map(|_| Vec::with_capacity(k_neighbors)).collect();
		(0..n_threads).into_par_iter()
		.zip(adjacency.par_chunks_mut(batch_per_thread))
		.zip(query.axis_chunks_iter(Axis(0), batch_per_thread).collect::<Vec<_>>().into_par_iter())
		// .into_iter()
		.for_each(|((i_thread, adjacency_chunk), query_chunk)| {
			let mut cache = self._new_search_cache(max_heap_size);
			let mut entrypoints_override = Vec::new();
			entrypoints_override.push(R::zero());
			let i_start = i_thread * batch_per_thread;
			let i_end = i_start + adjacency_chunk.len();
			(i_start..i_end)
			.zip(adjacency_chunk.iter_mut())
			.zip(query_chunk.axis_iter(Axis(0)))
			.for_each(|((i_q, adj), q)| {
				*entrypoints_override.get_mut(0).unwrap() = R::from(i_q).unwrap();
				self._init_cache(&mut cache, &q, k_neighbors, max_heap_size, Some(&entrypoints_override));
				self.greedy_search_layer_with_cache(
					&q,
					&mut cache,
					max_heap_size,
					0,
				);
				let (ids_i, dists_i) = cache.extract_nn(k_neighbors);
				dists_i.into_iter().zip(ids_i.into_iter()).for_each(|(w,i)| adj.push((w,i)))
			});
		});
		WDirLoLGraph {
			adjacency: adjacency,
			n_edges: self.n_rows() * k_neighbors,
		}
	}
}


macro_rules! base_index_impls(
	($t1: ident, $($tn: ident),+) => {
		base_index_impls!($t1);
		base_index_impls!($($tn),+);
	};
	($base_type: ident) => {
		impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> MatrixDataSource<F> for $base_type<R, F, Dist, Mat, G> {
			const SUPPORTS_ROW_VIEW: bool = Mat::SUPPORTS_ROW_VIEW;
			#[inline(always)]
			fn n_rows(&self) -> usize { self.data.n_rows() }
			#[inline(always)]
			fn n_cols(&self) -> usize { self.data.n_cols() }
			#[inline(always)]
			fn get_row(&self, i_row: usize) -> Array1<F> { self.data.get_row(i_row) }
			#[inline(always)]
			fn get_row_view(&self, i_row: usize) -> &[F] { self.data.get_row_view(i_row) }
			#[inline(always)]
			fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<F> { self.data.get_rows(i_rows) }
			#[inline(always)]
			fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<F> { self.data.get_rows_slice(i_row_from, i_row_to) }
		}
		impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> RangeIndex<R,F,Dist> for $base_type<R, F, Dist, Mat, G> {
			#[inline(always)]
			fn range_query<D: Data<Elem=F>>(&self, _query: &ArrayBase<D,Ix1>, _range: F) -> (Array1<R>, Array1<F>) {
				panic!("Not implemented");
			}
			#[inline(always)]
			fn range_query_batch<D: Data<Elem=F>>(&self, _query: &ArrayBase<D,Ix2>, _range: F) -> (Vec<Array1<R>>, Vec<Array1<F>>) {
				panic!("Not implemented");
			}
		}
		impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync, Mat: MatrixDataSource<F>+Sync, G: Graph<R>+Sync> KnnIndex<R,F,Dist> for $base_type<R, F, Dist, Mat, G> {
			#[inline(always)]
			fn knn_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D,Ix1>, k: usize) -> (Array1<R>, Array1<F>) {
				self.greedy_search(query, k, 2*k, &mut self._new_search_cache(2*k))
			}
			#[inline(always)]
			fn knn_query_batch<D: Data<Elem=F>>(&self, query: &ArrayBase<D,Ix2>, k: usize) -> (Array2<R>, Array2<F>) {
				self.greedy_search_batch(query, k, 2*k)
			}
		}
		impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> IndexedDistance<R,F,Dist> for $base_type<R, F, Dist, Mat, G> {
			#[inline(always)]
			fn distance<D1: Data<Elem=F>, D2: Data<Elem=F>>(&self, i: &ArrayBase<D1,Ix1>, j: &ArrayBase<D2,Ix1>) -> F {
				self.distance.dist(i, j)
			}
		}
	};
);
base_index_impls!(GreedySingleGraphIndex, GreedyCappedSingleGraphIndex, GreedyLayeredGraphIndex, GreedyCappedLayeredGraphIndex);

macro_rules! graph_index_default_funs(
	(layered capped) => {
		graph_index_default_funs!(layered);
		graph_index_default_funs!(capped);
		#[inline(always)]
		fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, entrypoints_override: Option<&Vec<R>>) {
			cache.clear();
			cache.reserve(max_heap_size, self.max_frontier_size);
			let heap = &mut cache.heap;
			let ids = self.get_global_layer_ids(self.layer_count()-1);
			if entrypoints_override.is_none() && self.entry_points.is_none() {
				if ids.is_some() {
					let ids = unsafe{ids.unwrap_unchecked()};
					random_unique_uint::<R>(ids.len(), k_neighbors).iter().for_each(|&v|
						heap.push(self.half_indexed_distance(ids[unsafe{v.to_usize().unwrap_unchecked()}], q), v)
					);
				} else {
					random_unique_uint::<R>(self.n_rows(), k_neighbors).iter().for_each(|&v|
						heap.push(self.half_indexed_distance(v, q), v)
					);
				}
			} else {
				let entry_points = unsafe{if entrypoints_override.is_none() {
					self.entry_points.as_ref().unwrap_unchecked()
				} else {
					entrypoints_override.as_ref().unwrap_unchecked()
				}};
				if ids.is_some() {
					let ids = unsafe{ids.unwrap_unchecked()};
					random_unique_uint::<u64>(entry_points.len(), k_neighbors).iter().for_each(|&v| {
						let v = unsafe{*entry_points.get_unchecked(v as usize)};
						let vglobal = unsafe {*ids.get_unchecked(v.to_usize().unwrap_unchecked())};
						heap.push(self.half_indexed_distance(vglobal, q), v)
					});
				} else {
					random_unique_uint::<u64>(entry_points.len(), k_neighbors).iter().for_each(|&v| {
						let v = unsafe{*entry_points.get_unchecked(v as usize)};
						heap.push(self.half_indexed_distance(v, q), v)
					});
				}
			}
		}
		#[inline(always)]
		fn _new_search_cache(&self, max_heap_size: usize) -> Self::SearchCache {
			Self::SearchCache::new(self.graphs.iter().map(|g| g.n_vertices()).collect(), max_heap_size, self.max_frontier_size)
		}
	};
	(layered uncapped) => {
		graph_index_default_funs!(layered);
		graph_index_default_funs!(uncapped);
		#[inline(always)]
		fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, entrypoints_override: Option<&Vec<R>>) {
			cache.clear();
			cache.reserve(max_heap_size);
			let heap = &mut cache.heap;
			let ids = self.get_global_layer_ids(self.layer_count()-1);
			if entrypoints_override.is_none() && self.entry_points.is_none() {
				if ids.is_some() {
					let ids = unsafe{ids.unwrap_unchecked()};
					random_unique_uint::<R>(ids.len(), k_neighbors).iter().for_each(|&v|
						heap.push(self.half_indexed_distance(ids[unsafe{v.to_usize().unwrap_unchecked()}], q), v)
					);
				} else {
					random_unique_uint::<R>(self.n_rows(), k_neighbors).iter().for_each(|&v|
						heap.push(self.half_indexed_distance(v, q), v)
					);
				}
			} else {
				let entry_points = unsafe{if entrypoints_override.is_none() {
					self.entry_points.as_ref().unwrap_unchecked()
				} else {
					entrypoints_override.as_ref().unwrap_unchecked()
				}};
				if ids.is_some() {
					let ids = unsafe{ids.unwrap_unchecked()};
					random_unique_uint::<u64>(entry_points.len(), k_neighbors).iter().for_each(|&v| {
						let v = unsafe{*entry_points.get_unchecked(v as usize)};
						let vglobal = unsafe {*ids.get_unchecked(v.to_usize().unwrap_unchecked())};
						heap.push(self.half_indexed_distance(vglobal, q), v)
					});
				} else {
					random_unique_uint::<u64>(entry_points.len(), k_neighbors).iter().for_each(|&v| {
						let v = unsafe{*entry_points.get_unchecked(v as usize)};
						heap.push(self.half_indexed_distance(v, q), v)
					});
				}
			}
		}
		#[inline(always)]
		fn _new_search_cache(&self, max_heap_size: usize) -> Self::SearchCache {
			Self::SearchCache::new(self.graphs.iter().map(|g| g.n_vertices()).collect(), max_heap_size)
		}
	};
	(single capped) => {
		graph_index_default_funs!(single);
		graph_index_default_funs!(capped);
		#[inline(always)]
		fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, entrypoints_override: Option<&Vec<R>>) {
			cache.clear();
			cache.reserve(max_heap_size, self.max_frontier_size);
			let heap = &mut cache.heap;
			if entrypoints_override.is_none() && self.entry_points.is_none() {
				random_unique_uint::<R>(self.n_rows(), k_neighbors).iter().for_each(|&v|
					heap.push(self.half_indexed_distance(v, q), v)
				);
			} else {
				let entry_points = unsafe{if entrypoints_override.is_none() {
					self.entry_points.as_ref().unwrap_unchecked()
				} else {
					entrypoints_override.as_ref().unwrap_unchecked()
				}};
				random_unique_uint::<u64>(entry_points.len(), k_neighbors).iter().for_each(|&v| {
					let v = unsafe{*entry_points.get_unchecked(v as usize)};
					heap.push(self.half_indexed_distance(v, q), v)
				});
			}
		}
		#[inline(always)]
		fn _new_search_cache(&self, max_heap_size: usize) -> Self::SearchCache {
			Self::SearchCache::new(vec![self.graph.n_vertices()], max_heap_size, self.max_frontier_size)
		}
	};
	(single uncapped) => {
		graph_index_default_funs!(single);
		graph_index_default_funs!(uncapped);
		#[inline(always)]
		fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, entrypoints_override: Option<&Vec<R>>) {
			cache.clear();
			cache.reserve(max_heap_size);
			let heap = &mut cache.heap;
			if entrypoints_override.is_none() && self.entry_points.is_none() {
					random_unique_uint::<R>(self.n_rows(), k_neighbors).iter().for_each(|&v|
						heap.push(self.half_indexed_distance(v, q), v)
					);
			} else {
				let entry_points = unsafe{if entrypoints_override.is_none() {
					self.entry_points.as_ref().unwrap_unchecked()
				} else {
					entrypoints_override.as_ref().unwrap_unchecked()
				}};
				random_unique_uint::<u64>(entry_points.len(), k_neighbors).iter().for_each(|&v| {
					let v = unsafe{*entry_points.get_unchecked(v as usize)};
					heap.push(self.half_indexed_distance(v, q), v)
				});
			}
		}
		#[inline(always)]
		fn _new_search_cache(&self, max_heap_size: usize) -> Self::SearchCache {
			Self::SearchCache::new(vec![self.graph.n_vertices()], max_heap_size)
		}
	};
	(single) => {
		#[inline(always)]
		fn layer_count(&self) -> usize { 1 }
		#[inline(always)]
		fn get_layer(&self, layer: usize) -> Result<&impl Graph<R>, NoSuchLayerError> { if layer==0 {Ok(&self.graph)} else {Err(NoSuchLayerError)} }
		#[inline(always)]
		fn get_global_layer_ids(&self, _layer: usize) -> Option<&Vec<R>> { None }
		#[inline(always)]
		fn get_local_layer_ids(&self, _layer: usize) -> Option<&Vec<R>> { None }
	};
	(layered) => {
		#[inline(always)]
		fn layer_count(&self) -> usize { self.graphs.len() }
		#[inline(always)]
		fn get_layer(&self, layer: usize) -> Result<&impl Graph<R>, NoSuchLayerError> { if layer==0 {Ok(&self.graphs[layer])} else {Err(NoSuchLayerError)} }
		#[inline(always)]
		fn get_global_layer_ids(&self, layer: usize) -> Option<&Vec<R>> { if layer>0 && layer<=self.global_layer_ids.len() {Some(&self.global_layer_ids[layer-1])} else {None} }
		#[inline(always)]
		fn get_local_layer_ids(&self, layer: usize) -> Option<&Vec<R>> { if layer>0 && layer<=self.local_layer_ids.len() {Some(&self.local_layer_ids[layer-1])} else {None} }
		#[inline(always)]
		fn greedy_search<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, cache: &mut Self::SearchCache) -> (Array1<R>, Array1<F>) {
			self._init_cache(cache, q, if self.graphs.len() == 1 {max_heap_size} else {self.higher_level_max_heap_size}, max_heap_size, None);
			/* Search all layers graph */
			self.greedy_search_with_cache(q, cache, max_heap_size);
			/* Extract the k nearest neighbors */
			cache.extract_nn(k_neighbors)
		}
		#[inline(always)]
		fn greedy_search_with_cache<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, cache: &mut Self::SearchCache, max_heap_size: usize) {
			for layer in (0..self.layer_count()).rev() {
				/* Get heap from the current layer */
				self.greedy_search_layer_with_cache(q, cache, if layer==0 {max_heap_size} else {self.higher_level_max_heap_size}, layer);
				/* Apply local ID map if available */
				let idx_map = self.get_local_layer_ids(layer);
				if idx_map.is_some() {
					cache.apply_local_id_map(unsafe{idx_map.unwrap_unchecked()});
				}
			}
		}
	};
	(capped) => {
		type SearchCache = DefaultCappedSearchCache<R,F>;
	};
	(uncapped) => {
		type SearchCache = DefaultSearchCache<R,F>;
	};
);

pub struct GreedySingleGraphIndex<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> {
	_phantom: std::marker::PhantomData<(R,F)>,
	data: Mat,
	graph: G,
	distance: Dist,
	entry_points: Option<Vec<R>>,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> GreedySingleGraphIndex<R, F, Dist, Mat, G> {
	#[inline(always)]
	pub fn new(data: Mat, graph: G, distance: Dist, entry_points: Option<Vec<R>>) -> Self {
		Self{
			_phantom: std::marker::PhantomData,
			data,
			graph,
			distance,
			entry_points,
		}
	}
	#[inline(always)]
	pub fn n_edges(&self) -> usize { self.graph.n_edges() }
	#[inline(always)]
	pub fn graph(&self) -> &G { &self.graph }
	#[inline(always)]
	pub fn graph_mut(&mut self) -> &mut G { &mut self.graph }
	#[inline(always)]
	pub fn into_capped(self, max_frontier_size: usize) -> GreedyCappedSingleGraphIndex<R, F, Dist, Mat, G> {
		GreedyCappedSingleGraphIndex::new(self.data, self.graph, self.distance, max_frontier_size, self.entry_points)
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync, Mat: MatrixDataSource<F>+Sync, G: Graph<R>+Sync> GraphIndex<R, F, Dist> for GreedySingleGraphIndex<R, F, Dist, Mat, G> {
	graph_index_default_funs!(single uncapped);
	fn greedy_search_layer_with_cache<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, cache: &mut Self::SearchCache, max_heap_size: usize, _layer: usize) {
		let visited_set = unsafe{cache.visited_sets.get_unchecked_mut(0)};
		visited_set.clear();
		let frontier = &mut cache.frontier;
		frontier.clear();
		let heap = &mut cache.heap;
		heap.reserve(max_heap_size);
		heap.iter().for_each(|&(d,v)| {
			frontier.push(d, v);
			visited_set.insert(v);
		});
		let graph = &self.graph;
		while let Some((d, v)) = frontier.pop() {
			if d > heap.peek().unwrap().0 { break; }
			// if visited_set.contains(&v) { continue; }
			graph.foreach_neighbor(v, |&i| {
				if visited_set.insert(i) {
					let neighbor_dist = self.half_indexed_distance(i, q);
					if heap.size() < max_heap_size {
						heap.push(neighbor_dist, i);
					} else if heap.peek().unwrap().0 > neighbor_dist {
						heap.pop();
						heap.push(neighbor_dist, i);
					}
					frontier.push(neighbor_dist, i);
				}
			});
		}
	}
}

pub struct GreedyCappedSingleGraphIndex<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> {
	_phantom: std::marker::PhantomData<(R,F)>,
	data: Mat,
	graph: G,
	distance: Dist,
	pub max_frontier_size: usize,
	entry_points: Option<Vec<R>>,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> GreedyCappedSingleGraphIndex<R, F, Dist, Mat, G> {
	#[inline(always)]
	pub fn new(data: Mat, graph: G, distance: Dist, max_frontier_size: usize, entry_points: Option<Vec<R>>) -> Self {
		Self{
			_phantom: std::marker::PhantomData,
			data,
			graph,
			distance,
			max_frontier_size,
			entry_points,
		}
	}
	#[inline(always)]
	pub fn n_edges(&self) -> usize { self.graph.n_edges() }
	#[inline(always)]
	pub fn max_frontier_size(&self) -> usize { self.max_frontier_size }
	#[inline(always)]
	pub fn set_max_frontier_size(&mut self, max_frontier_size: usize) { self.max_frontier_size = max_frontier_size; }
	#[inline(always)]
	pub fn graph(&self) -> &G { &self.graph }
	#[inline(always)]
	pub fn graph_mut(&mut self) -> &mut G { &mut self.graph }
	#[inline(always)]
	pub fn into_uncapped(self) -> GreedySingleGraphIndex<R, F, Dist, Mat, G> {
		GreedySingleGraphIndex::new(self.data, self.graph, self.distance, self.entry_points)
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync, Mat: MatrixDataSource<F>+Sync, G: Graph<R>+Sync> GraphIndex<R, F, Dist> for GreedyCappedSingleGraphIndex<R, F, Dist, Mat, G> {
	graph_index_default_funs!(single capped);
	fn greedy_search_layer_with_cache<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, cache: &mut Self::SearchCache, max_heap_size: usize, _layer: usize) {
		let visited_set = unsafe{cache.visited_sets.get_unchecked_mut(0)};
		visited_set.clear();
		let frontier = &mut cache.frontier;
		frontier.clear();
		let heap = &mut cache.heap;
		heap.reserve(max_heap_size);
		heap.iter().for_each(|&(d,v)| {
			if frontier.size() < self.max_frontier_size {
				frontier.push(d, v);
			} else {
				frontier.push_pop::<false>(d, v);
			}
			visited_set.insert(v);
		});
		let graph = &self.graph;
		while let Some((d, v)) = frontier.pop::<true>() {
			if d > heap.peek().unwrap().0 { break; }
			graph.foreach_neighbor(v, |&i| {
				if visited_set.insert(i) {
					let neighbor_dist = self.half_indexed_distance(i, q);
					if heap.size() < max_heap_size {
						heap.push(neighbor_dist, i);
					} else if heap.peek().unwrap().0 > neighbor_dist {
						heap.pop();
						heap.push(neighbor_dist, i);
					}
					if frontier.size() < self.max_frontier_size {
						frontier.push(neighbor_dist, i);
					} else {
						frontier.push_pop::<false>(neighbor_dist, i);
					}
				}
			});
		}
	}
}





pub struct GreedyLayeredGraphIndex<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> {
	_phantom: std::marker::PhantomData<(R,F)>,
	data: Mat,
	graphs: Vec<G>,
	local_layer_ids: Vec<Vec<R>>,
	global_layer_ids: Vec<Vec<R>>,
	distance: Dist,
	higher_level_max_heap_size: usize,
	entry_points: Option<Vec<R>>,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> GreedyLayeredGraphIndex<R, F, Dist, Mat, G> {
	#[inline(always)]
	pub fn new(data: Mat, graphs: Vec<G>, local_layer_ids: Vec<Vec<R>>, global_layer_ids: Vec<Vec<R>>, distance: Dist, higher_level_max_heap_size: usize, top_entry_points: Option<Vec<R>>) -> Self {
		Self{
			_phantom: std::marker::PhantomData,
			data,
			graphs,
			local_layer_ids,
			global_layer_ids,
			distance,
			higher_level_max_heap_size,
			entry_points: top_entry_points,
		}
	}
	#[inline(always)]
	pub fn n_edges(&self, layer: usize) -> Option<usize> { if layer >= self.graphs.len() {None} else {Some(self.graphs[layer].n_edges())} }
	#[inline(always)]
	pub fn higher_level_max_heap_size(&self) -> usize { self.higher_level_max_heap_size }
	#[inline(always)]
	pub fn set_higher_level_max_heap_size(&mut self, higher_level_max_heap_size: usize) { self.higher_level_max_heap_size = higher_level_max_heap_size; }
	#[inline(always)]
	pub fn graphs(&self) -> &Vec<G> { &self.graphs }
	#[inline(always)]
	pub fn graphs_mut(&mut self) -> &mut Vec<G> { &mut self.graphs }
	#[inline(always)]
	pub fn into_capped(self, max_frontier_size: usize) -> GreedyCappedLayeredGraphIndex<R, F, Dist, Mat, G> {
		GreedyCappedLayeredGraphIndex::new(self.data, self.graphs, self.local_layer_ids, self.global_layer_ids, self.distance, self.higher_level_max_heap_size, max_frontier_size, self.entry_points.clone())
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync, Mat: MatrixDataSource<F>+Sync, G: Graph<R>+Sync> GraphIndex<R, F, Dist> for GreedyLayeredGraphIndex<R, F, Dist, Mat, G> {
	graph_index_default_funs!(layered uncapped);
	fn greedy_search_layer_with_cache<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, cache: &mut Self::SearchCache, max_heap_size: usize, layer: usize) {
		let visited_set = unsafe{cache.visited_sets.get_unchecked_mut(layer)};
		visited_set.clear();
		let frontier = &mut cache.frontier;
		frontier.clear();
		let heap = &mut cache.heap;
		heap.reserve(max_heap_size);
		heap.iter().for_each(|&(d,v)| {
			frontier.push(d, v);
			visited_set.insert(v);
		});
		unsafe {
			let graph = self.graphs.get_unchecked(layer);
			let global_ids = self.get_global_layer_ids(layer);
			if global_ids.is_none() {
				while let Some((d, v)) = frontier.pop() {
					if d > heap.peek().unwrap_unchecked().0 { break; }
					graph.foreach_neighbor(v, |&i| {
						if visited_set.insert(i) {
							let neighbor_dist = self.half_indexed_distance(i, q);
							if heap.size() < max_heap_size {
								heap.push(neighbor_dist, i);
							} else if heap.peek().unwrap_unchecked().0 > neighbor_dist {
								heap.pop();
								heap.push(neighbor_dist, i);
							}
							frontier.push(neighbor_dist, i);
						}
					});
				}
			} else {
				let global_ids = global_ids.unwrap_unchecked();
				let to_global = |i: R| global_ids[i.to_usize().unwrap_unchecked()];
				while let Some((d, v)) = frontier.pop() {
					if d > heap.peek().unwrap_unchecked().0 { break; }
					graph.foreach_neighbor(v, |&i| {
						if visited_set.insert(i) {
							let neighbor_dist = self.half_indexed_distance(to_global(i), q);
							if heap.size() < max_heap_size {
								heap.push(neighbor_dist, i);
							} else if heap.peek().unwrap_unchecked().0 > neighbor_dist {
								heap.pop();
								heap.push(neighbor_dist, i);
							}
							frontier.push(neighbor_dist, i);
						}
					});
				}
			}
		}
	}
}

pub struct GreedyCappedLayeredGraphIndex<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> {
	_phantom: std::marker::PhantomData<(R,F)>,
	data: Mat,
	graphs: Vec<G>,
	local_layer_ids: Vec<Vec<R>>,
	global_layer_ids: Vec<Vec<R>>,
	distance: Dist,
	pub max_frontier_size: usize,
	higher_level_max_heap_size: usize,
	entry_points: Option<Vec<R>>,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> GreedyCappedLayeredGraphIndex<R, F, Dist, Mat, G> {
	#[inline(always)]
	pub fn new(data: Mat, graphs: Vec<G>, local_layer_ids: Vec<Vec<R>>, global_layer_ids: Vec<Vec<R>>, distance: Dist, higher_level_max_heap_size: usize, max_frontier_size: usize, top_entry_points: Option<Vec<R>>) -> Self {
		Self{
			_phantom: std::marker::PhantomData,
			data,
			graphs,
			local_layer_ids,
			global_layer_ids,
			distance,
			max_frontier_size,
			higher_level_max_heap_size,
			entry_points: top_entry_points,
		}
	}
	#[inline(always)]
	pub fn n_edges(&self, layer: usize) -> Option<usize> { if layer >= self.graphs.len() {None} else {Some(self.graphs[layer].n_edges())} }
	#[inline(always)]
	pub fn higher_level_max_heap_size(&self) -> usize { self.higher_level_max_heap_size }
	#[inline(always)]
	pub fn set_higher_level_max_heap_size(&mut self, higher_level_max_heap_size: usize) { self.higher_level_max_heap_size = higher_level_max_heap_size; }
	#[inline(always)]
	pub fn max_frontier_size(&self) -> usize { self.max_frontier_size }
	#[inline(always)]
	pub fn set_max_frontier_size(&mut self, max_frontier_size: usize) { self.max_frontier_size = max_frontier_size; }
	#[inline(always)]
	pub fn graphs(&self) -> &Vec<G> { &self.graphs }
	#[inline(always)]
	pub fn graphs_mut(&mut self) -> &mut Vec<G> { &mut self.graphs }
	#[inline(always)]
	pub fn into_uncapped(self) -> GreedyLayeredGraphIndex<R, F, Dist, Mat, G> {
		GreedyLayeredGraphIndex::new(self.data, self.graphs, self.local_layer_ids, self.global_layer_ids, self.distance, self.higher_level_max_heap_size, self.entry_points.clone())
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync, Mat: MatrixDataSource<F>+Sync, G: Graph<R>+Sync> GraphIndex<R, F, Dist> for GreedyCappedLayeredGraphIndex<R, F, Dist, Mat, G> {
	graph_index_default_funs!(layered capped);
	fn greedy_search_layer_with_cache<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, cache: &mut Self::SearchCache, max_heap_size: usize, layer: usize) {
		let visited_set = unsafe{cache.visited_sets.get_unchecked_mut(layer)};
		visited_set.clear();
		let frontier = &mut cache.frontier;
		frontier.clear();
		let heap = &mut cache.heap;
		heap.reserve(max_heap_size);
		heap.iter().for_each(|&(d,v)| {
			frontier.push(d, v);
			visited_set.insert(v);
		});
		let graph = &self.graphs[layer];
		let global_ids = self.get_global_layer_ids(layer);
		if global_ids.is_none() {
			while let Some((d, v)) = frontier.pop::<true>() {
				if d > heap.peek().unwrap().0 { break; }
				// if visited_set.contains(&v) { continue; }
				graph.foreach_neighbor(v, |&i| {
					if visited_set.insert(i) {
						let neighbor_dist = self.half_indexed_distance(i, q);
						if heap.size() < max_heap_size {
							heap.push(neighbor_dist, i);
						} else if heap.peek().unwrap().0 > neighbor_dist {
							heap.pop();
							heap.push(neighbor_dist, i);
						}
						if frontier.size() < self.max_frontier_size {
							frontier.push(neighbor_dist, i);
						} else {
							frontier.push_pop::<false>(neighbor_dist, i);
						}
					}
				});
			}
		} else {
			let global_ids = unsafe{global_ids.unwrap_unchecked()};
			let to_global = |i: R| unsafe{ global_ids[i.to_usize().unwrap_unchecked()] };
			while let Some((d, v)) = frontier.pop::<true>() {
				if d > heap.peek().unwrap().0 { break; }
				graph.foreach_neighbor(v, |&i| {
					if visited_set.insert(i) {
						let neighbor_dist = self.half_indexed_distance(to_global(i), q);
						if heap.size() < max_heap_size {
							heap.push(neighbor_dist, i);
						} else if heap.peek().unwrap().0 > neighbor_dist {
							heap.pop();
							heap.push(neighbor_dist, i);
						}
						if frontier.size() < self.max_frontier_size {
							frontier.push(neighbor_dist, i);
						} else {
							frontier.push_pop::<false>(neighbor_dist, i);
						}
					}
				});
			}
		}
	}
}


