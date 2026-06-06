use core::panic;

use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, s};
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
	fn range_query_batch<M: MatrixDataSource<F>+Sync>(&self, _query: &M, _range: F) -> (Vec<Array1<R>>, Vec<Array1<F>>);
}
pub trait KnnIndex<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>> {
	fn knn_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix1>, k: usize) -> (Array1<R>, Array1<F>);
	fn knn_query_batch<M: MatrixDataSource<F>+Sync>(&self, query: &M, k: usize) -> (Array2<R>, Array2<F>);
}
trait_combiner!(GeneralIndex[R: SyncUnsignedInteger, F: SyncFloat, Dist: (Distance<F>)]: (RangeIndex<R, F, Dist>) + (KnnIndex<R, F, Dist>) + (IndexedDistance<R, F, Dist>) + (MatrixDataSource<F>));



pub fn bruteforce_neighbors<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	Dist: Distance<F>+Sync,
	DM: MatrixDataSource<F>+Sync,
	QM: MatrixDataSource<F>+Sync,
>(data: &DM, queries: &QM, dist: &Dist, k: usize) -> (Array2<R>, Array2<F>) {
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
		.enumerate()
		.par_bridge()
		.for_each(|(i_chunk, (mut id_chunk,mut dist_chunk))| {
			let chunk_offset = i_chunk + chunk_size;
			let mut dist_cache = Vec::with_capacity(nd);
			id_chunk.axis_iter_mut(Axis(0))
			.zip(dist_chunk.axis_iter_mut(Axis(0)))
			.enumerate()
			.for_each(|(i_query,(mut ids_target, mut dists_target))| {
				let i_query = i_query + chunk_offset;
				let iq = queries.get_abs_row(i_query);
				dist_cache.clear();
				(0..data.n_rows()).into_iter()
				.for_each(|i_data| {
					let ix = data.get_abs_row(i_data);
					dist_cache.push((i_data,dist.dist_slice(iq.as_slice(), ix.as_slice())));
				});
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
	fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, entry_layer: Option<usize>, entrypoints_override: Option<&Vec<R>>);
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
		self._init_cache(cache, q, k_neighbors, max_heap_size, None, None);
		/* Search all layers graph */
		self.greedy_search_with_cache(q, cache, max_heap_size);
		/* Extract the k nearest neighbors */
		cache.extract_nn(k_neighbors)
	}
	fn greedy_search_batch<M: MatrixDataSource<F>+Sync>(&self, q: &M, k_neighbors: usize, max_heap_size: usize) -> (Array2<R>, Array2<F>) {
		let mut ids = Array2::from_elem((q.n_rows(), k_neighbors), R::zero());
		let mut dists = Array2::from_elem((q.n_rows(), k_neighbors), F::zero());
		let n_threads = rayon::current_num_threads();
		let n_queries = q.n_rows();
		let batch_per_thread = (n_queries+n_threads-1)/n_threads;
		let raw_iter = ids.axis_chunks_iter_mut(Axis(0), batch_per_thread)
		.zip(dists.axis_chunks_iter_mut(Axis(0), batch_per_thread))
		.enumerate()
		.map(|(a,(b,c))|(a,b,c)).collect::<Vec<_>>();
		raw_iter
		.into_par_iter()
		// .into_iter()
		.for_each(|(i_chunk, mut id_chunk, mut dist_chunk)| {
			let chunk_offset = i_chunk * batch_per_thread;
			let mut cache = self._new_search_cache(max_heap_size);
			id_chunk.axis_iter_mut(Axis(0))
			.zip(dist_chunk.axis_iter_mut(Axis(0)))
			.enumerate()
			.for_each(|(i_query, (mut ids, mut dists))| {
				let i_query = i_query + chunk_offset;
				let iq = q.get_abs_row(i_query);
				/* Fixme: This should ideally reuse the same heap memory for each search within a thread */
				let (ids_i, dists_i) = self.greedy_search(&iq.as_view(), k_neighbors, max_heap_size, &mut cache);
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
	/// Self join using the regular `greedy_search_batch` function returning a weighted graph from the neighbors
	fn self_join_query(&self, k_neighbors: usize, max_heap_size: usize) -> WDirLoLGraph<R,F> {
		self.self_join_query_slice(k_neighbors, max_heap_size, None)
	}
	fn self_join_query_slice(&self, k_neighbors: usize, max_heap_size: usize, slice: Option<(usize,usize)>) -> WDirLoLGraph<R,F> {
		let (start,end) = slice.unwrap_or((0,self.n_rows()));
		let n_rows = end-start;
		/* TODO: If row slice view is not available, but row view is, the parallelization should use row views instead */
		let (ids, dists) = if Self::SUPPORTS_ROW_SLICE_VIEW {
			let query = self.get_row_slice_view(start, end);
			self.greedy_search_batch(&query, k_neighbors+1, max_heap_size)
		} else {
			let query = self.get_rows_slice(start, end);
			self.greedy_search_batch(&query, k_neighbors+1, max_heap_size)
		};
		let n_threads = rayon::current_num_threads();
		let batch_per_thread = (n_rows+n_threads-1)/n_threads;
		let mut adjacency = Vec::with_capacity(n_rows);
		(start..end).for_each(|_| adjacency.push(Vec::with_capacity(k_neighbors)));
		adjacency.chunks_mut(batch_per_thread)
		.zip(ids.axis_chunks_iter(Axis(0), batch_per_thread))
		.zip(dists.axis_chunks_iter(Axis(0), batch_per_thread))
		.enumerate()
		.par_bridge()
		.map(|(a,((b,c),d))| (a,b,c,d))
		.for_each(|(i_chunk, adj_chunk, ids_chunk, dists_chunk)| {
			let chunk_offset = start + i_chunk * batch_per_thread;
			adj_chunk.iter_mut()
			.zip(ids_chunk.axis_iter(Axis(0)))
			.zip(dists_chunk.axis_iter(Axis(0)))
			.enumerate()
			.map(|(a,((b,c),d))| (a,b,c,d))
			.for_each(|(i_sample,adj,ids,dists)| {
				let i_self = chunk_offset + i_sample;
				dists.into_iter()
				.zip(ids.into_iter())
				.filter(|(_, &i)| i.to_usize().unwrap() != i_self)
				.take(k_neighbors)
				.for_each(|(&w,&i)| adj.push((w,i)));
			});
		});
		WDirLoLGraph {
			adjacency: adjacency,
			n_edges: n_rows * k_neighbors,
		}
	}
	/// Self join using the `greedy_search_layer_with_cache` function on the bottom layer returning a weighted graph from the neighbors
	fn self_join_query_local(&self, k_neighbors: usize, max_heap_size: usize) -> WDirLoLGraph<R,F> {
		self.self_join_query_local_slice(k_neighbors, max_heap_size, None)
	}
	fn self_join_query_local_slice(&self, k_neighbors: usize, max_heap_size: usize, slice: Option<(usize,usize)>) -> WDirLoLGraph<R,F> {
		let (start,end) = slice.unwrap_or((0,self.n_rows()));
		let n_queries = end-start;
		let n_threads = rayon::current_num_threads();
		let batch_per_thread = (n_queries+n_threads-1)/n_threads;
		let mut adjacency: Vec<Vec<(F,R)>> = (start..end).map(|_| Vec::with_capacity(k_neighbors)).collect();
		(0..n_threads).into_par_iter()
		.zip(adjacency.par_chunks_mut(batch_per_thread))
		.for_each(|(i_thread, adjacency_chunk)| {
			let mut cache = self._new_search_cache(max_heap_size);
			let mut entrypoints_override = Vec::new();
			entrypoints_override.push(R::zero());
			let i_start = start + i_thread * batch_per_thread;
			let i_end = i_start + adjacency_chunk.len();
			(i_start..i_end)
			.zip(adjacency_chunk.iter_mut())
			.for_each(|(i_q, adj)| {
				*entrypoints_override.get_mut(0).unwrap() = R::from(i_q).unwrap();
				let q = self.get_abs_row(i_q);
				let q_view = q.as_view();
				self._init_cache(&mut cache, &q_view, k_neighbors+1, max_heap_size, Some(0), Some(&entrypoints_override));
				self.greedy_search_layer_with_cache(
					&q_view,
					&mut cache,
					max_heap_size,
					0,
				);
				let (ids_i, dists_i) = cache.extract_nn(k_neighbors+1);
				dists_i.into_iter().zip(ids_i.into_iter())
				.filter(|(_,i)| i.to_usize().unwrap() != i_q)
				.take(k_neighbors)
				.for_each(|(w,i)| adj.push((w,i)))
			});
		});
		WDirLoLGraph {
			adjacency: adjacency,
			n_edges: self.n_rows() * k_neighbors,
		}
	}
	/// Self join using the regular `greedy_search_batch` function returning the neighbors as index and distance arrays
	fn self_join_query_arr(&self, k_neighbors: usize, max_heap_size: usize) -> (Array2<R>, Array2<F>) {
		self.self_join_query_arr_slice(k_neighbors, max_heap_size, None)
	}
	fn self_join_query_arr_slice(&self, k_neighbors: usize, max_heap_size: usize, slice: Option<(usize,usize)>) -> (Array2<R>, Array2<F>) {
		let (start,end) = slice.unwrap_or((0,self.n_rows()));
		let n_rows = end-start;
		/* Make the actual query to get the neighbors of the self join with k+1 neighbors in case the query is itself is found */
		/* TODO: If row slice view is not available, but row view is, the parallelization should use row views instead */
		let (ids, dists) = if Self::SUPPORTS_ROW_SLICE_VIEW {
			let query = self.get_row_slice_view(start,end);
			self.greedy_search_batch(&query, k_neighbors+1, max_heap_size)
		} else {
			let query = self.get_rows_slice(start,end);
			self.greedy_search_batch(&query, k_neighbors+1, max_heap_size)
		};
		/* Remove each item if it is included in its own query result */
		let n_threads = rayon::current_num_threads();
		let batch_per_thread = (n_rows+n_threads-1)/n_threads;
		let mut ids_out = Array2::zeros((n_rows,k_neighbors));
		let mut dists_out = Array2::zeros((n_rows,k_neighbors));
		ids.axis_chunks_iter(Axis(0), batch_per_thread)
		.zip(dists.axis_chunks_iter(Axis(0), batch_per_thread))
		.zip(ids_out.axis_chunks_iter_mut(Axis(0), batch_per_thread))
		.zip(dists_out.axis_chunks_iter_mut(Axis(0), batch_per_thread))
		.enumerate()
		.par_bridge()
		.map(|(a,(((b,c),d),e))| (a,b,c,d,e))
		.for_each(|(i_chunk, ids_chunk, dists_chunk, mut ids_out_chunk, mut dists_out_chunk)| {
			let chunk_offset = start + i_chunk * batch_per_thread;
			ids_chunk.axis_iter(Axis(0))
			.zip(dists_chunk.axis_iter(Axis(0)))
			.zip(ids_out_chunk.axis_iter_mut(Axis(0)))
			.zip(dists_out_chunk.axis_iter_mut(Axis(0)))
			.enumerate()
			.map(|(a,(((b,c),d),e))| (a,b,c,d,e))
			.for_each(|(i_sample, ids, dists, mut ids_out, mut dists_out)| {
				let i_self = chunk_offset + i_sample;
				if ids[0].to_usize().unwrap() == i_self {
					ids_out.assign(&ids.slice(s![1..]));
					dists_out.assign(&dists.slice(s![1..]));
				} else {
					ids_out.assign(&ids.slice(s![..k_neighbors]));
					dists_out.assign(&dists.slice(s![..k_neighbors]));
				}
			});
		});
		/* Return the cropped result */
		(ids_out, dists_out)
	}
	/// Self join using the `greedy_search_layer_with_cache` function on the bottom layer returning the neighbors as index and distance arrays
	fn self_join_query_local_arr(&self, k_neighbors: usize, max_heap_size: usize) -> (Array2<R>, Array2<F>) {
		self.self_join_query_local_arr_slice(k_neighbors, max_heap_size, None)
	}
	fn self_join_query_local_arr_slice(&self, k_neighbors: usize, max_heap_size: usize, slice: Option<(usize,usize)>) -> (Array2<R>, Array2<F>) {
		let (start,end) = slice.unwrap_or((0,self.n_rows()));
		let n_queries = end-start;
		let n_threads = rayon::current_num_threads();
		let batch_per_thread = (n_queries+n_threads-1)/n_threads;
		let mut ids_out = Array2::zeros((n_queries,k_neighbors));
		let mut dists_out = Array2::zeros((n_queries,k_neighbors));
		ids_out.axis_chunks_iter_mut(Axis(0), batch_per_thread)
		.zip(dists_out.axis_chunks_iter_mut(Axis(0), batch_per_thread))
		.enumerate()
		.par_bridge()
		.map(|(a,(b,c))| (a,b,c))
		.for_each(|(i_chunk, mut ids_chunk, mut dists_chunk)| {
			let chunk_offset = start + i_chunk * batch_per_thread;
			let mut cache = self._new_search_cache(max_heap_size);
			let mut entrypoints_override = Vec::new();
			entrypoints_override.push(R::zero());
			ids_chunk.axis_iter_mut(Axis(0))
			.zip(dists_chunk.axis_iter_mut(Axis(0)))
			.enumerate()
			.map(|(a,(b,c))| (a,b,c))
			.for_each(|(i_sample, mut ids, mut dists)| {
				let i_q = chunk_offset + i_sample;
				*entrypoints_override.get_mut(0).unwrap() = R::from(i_q).unwrap();
				let iq = self.get_abs_row(i_q);
				let iq_view = iq.as_view();
				self._init_cache(&mut cache, &iq_view, k_neighbors+1, max_heap_size, Some(0), Some(&entrypoints_override));
				self.greedy_search_layer_with_cache(
					&iq_view,
					&mut cache,
					max_heap_size,
					0,
				);
				let (ids_i, dists_i) = cache.extract_nn(k_neighbors+1);
				dists_i.into_iter().zip(ids_i.into_iter())
				.filter(|(_,i)| i.to_usize().unwrap() != i_q)
				.take(k_neighbors)
				.zip(dists.iter_mut().zip(ids.iter_mut()))
				.for_each(|((w,i),(w_out,i_out))| {
					*w_out = w;
					*i_out = i;
				});
			});
		});
		(ids_out, dists_out)
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
			const SUPPORTS_ROW_SLICE_VIEW: bool = Mat::SUPPORTS_ROW_SLICE_VIEW;
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
			#[inline(always)]
			fn get_row_slice_view(&self, i_row_from: usize, i_row_to: usize) -> ArrayView2<F> { self.data.get_row_slice_view(i_row_from, i_row_to) }
		}
		impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> RangeIndex<R,F,Dist> for $base_type<R, F, Dist, Mat, G> {
			#[inline(always)]
			fn range_query<D: Data<Elem=F>>(&self, _query: &ArrayBase<D,Ix1>, _range: F) -> (Array1<R>, Array1<F>) {
				panic!("Not implemented");
			}
			#[inline(always)]
			fn range_query_batch<M: MatrixDataSource<F>+Sync>(&self, _query: &M, _range: F) -> (Vec<Array1<R>>, Vec<Array1<F>>) {
				panic!("Not implemented");
			}
		}
		impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync, Mat: MatrixDataSource<F>+Sync, G: Graph<R>+Sync> KnnIndex<R,F,Dist> for $base_type<R, F, Dist, Mat, G> {
			#[inline(always)]
			fn knn_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D,Ix1>, k: usize) -> (Array1<R>, Array1<F>) {
				self.greedy_search(query, k, 2*k, &mut self._new_search_cache(2*k))
			}
			#[inline(always)]
			fn knn_query_batch<M: MatrixDataSource<F>+Sync>(&self, query: &M, k: usize) -> (Array2<R>, Array2<F>) {
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
		fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, entry_layer: Option<usize>, entrypoints_override: Option<&Vec<R>>) {
			cache.clear();
			cache.reserve(max_heap_size, self.max_frontier_size);
			let heap = &mut cache.heap;
			let entry_layer = entry_layer.unwrap_or(self.layer_count()-1);
			let ids = self.get_global_layer_ids(entry_layer);
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
		fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, entry_layer: Option<usize>, entrypoints_override: Option<&Vec<R>>) {
			cache.clear();
			cache.reserve(max_heap_size);
			let heap = &mut cache.heap;
			let entry_layer = entry_layer.unwrap_or(self.layer_count()-1);
			let ids = self.get_global_layer_ids(entry_layer);
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
		fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, _entry_layer: Option<usize>, entrypoints_override: Option<&Vec<R>>) {
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
		fn _init_cache<D: Data<Elem=F>>(&self, cache: &mut Self::SearchCache, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, _entry_layer: Option<usize>, entrypoints_override: Option<&Vec<R>>) {
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
			self._init_cache(cache, q, if self.graphs.len() == 1 {max_heap_size} else {self.higher_level_max_heap_size}, max_heap_size, None, None);
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
	pub _phantom: std::marker::PhantomData<(R,F)>,
	pub data: Mat,
	pub graph: G,
	pub distance: Dist,
	pub entry_points: Option<Vec<R>>,
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
	pub fn with_distance<DistNew: Distance<F>>(self, dist: DistNew) -> GreedySingleGraphIndex<R,F,DistNew,Mat,G> {
		GreedySingleGraphIndex {
			_phantom: self._phantom,
			data: self.data,
			graph: self.graph,
			distance: dist,
			entry_points: self.entry_points,
		}
	}
	pub fn with_distance_and_data<DistNew: Distance<F>, MatNew: MatrixDataSource<F>>(self, dist: DistNew, data: MatNew) -> GreedySingleGraphIndex<R,F,DistNew,MatNew,G> {
		GreedySingleGraphIndex {
			_phantom: self._phantom,
			data: data,
			graph: self.graph,
			distance: dist,
			entry_points: self.entry_points,
		}
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
	pub _phantom: std::marker::PhantomData<(R,F)>,
	pub data: Mat,
	pub graph: G,
	pub distance: Dist,
	pub max_frontier_size: usize,
	pub entry_points: Option<Vec<R>>,
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
	pub fn with_distance<DistNew: Distance<F>>(self, dist: DistNew) -> GreedyCappedSingleGraphIndex<R,F,DistNew,Mat,G> {
		GreedyCappedSingleGraphIndex {
			_phantom: self._phantom,
			data: self.data,
			graph: self.graph,
			distance: dist,
			max_frontier_size: self.max_frontier_size,
			entry_points: self.entry_points,
		}
	}
	pub fn with_distance_and_data<DistNew: Distance<F>, MatNew: MatrixDataSource<F>>(self, dist: DistNew, data: MatNew) -> GreedyCappedSingleGraphIndex<R,F,DistNew,MatNew,G> {
		GreedyCappedSingleGraphIndex {
			_phantom: self._phantom,
			data: data,
			graph: self.graph,
			distance: dist,
			max_frontier_size: self.max_frontier_size,
			entry_points: self.entry_points,
		}
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
	pub _phantom: std::marker::PhantomData<(R,F)>,
	pub data: Mat,
	pub graphs: Vec<G>,
	pub local_layer_ids: Vec<Vec<R>>,
	pub global_layer_ids: Vec<Vec<R>>,
	pub distance: Dist,
	pub higher_level_max_heap_size: usize,
	pub entry_points: Option<Vec<R>>,
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
	pub fn with_distance<DistNew: Distance<F>>(self, dist: DistNew) -> GreedyLayeredGraphIndex<R,F,DistNew,Mat,G> {
		GreedyLayeredGraphIndex {
			_phantom: self._phantom,
			data: self.data,
			graphs: self.graphs,
			local_layer_ids: self.local_layer_ids,
			global_layer_ids: self.global_layer_ids,
			distance: dist,
			higher_level_max_heap_size: self.higher_level_max_heap_size,
			entry_points: self.entry_points,
		}
	}
	pub fn with_distance_and_data<DistNew: Distance<F>, MatNew: MatrixDataSource<F>>(self, dist: DistNew, data: MatNew) -> GreedyLayeredGraphIndex<R,F,DistNew,MatNew,G> {
		GreedyLayeredGraphIndex {
			_phantom: self._phantom,
			data: data,
			graphs: self.graphs,
			local_layer_ids: self.local_layer_ids,
			global_layer_ids: self.global_layer_ids,
			distance: dist,
			higher_level_max_heap_size: self.higher_level_max_heap_size,
			entry_points: self.entry_points,
		}
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
	pub _phantom: std::marker::PhantomData<(R,F)>,
	pub data: Mat,
	pub graphs: Vec<G>,
	pub local_layer_ids: Vec<Vec<R>>,
	pub global_layer_ids: Vec<Vec<R>>,
	pub distance: Dist,
	pub max_frontier_size: usize,
	pub higher_level_max_heap_size: usize,
	pub entry_points: Option<Vec<R>>,
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
	pub fn with_distance<DistNew: Distance<F>>(self, dist: DistNew) -> GreedyCappedLayeredGraphIndex<R,F,DistNew,Mat,G> {
		GreedyCappedLayeredGraphIndex {
			_phantom: self._phantom,
			data: self.data,
			graphs: self.graphs,
			local_layer_ids: self.local_layer_ids,
			global_layer_ids: self.global_layer_ids,
			distance: dist,
			max_frontier_size: self.max_frontier_size,
			higher_level_max_heap_size: self.higher_level_max_heap_size,
			entry_points: self.entry_points,
		}
	}
	pub fn with_distance_and_data<DistNew: Distance<F>, MatNew: MatrixDataSource<F>>(self, dist: DistNew, data: MatNew) -> GreedyCappedLayeredGraphIndex<R,F,DistNew,MatNew,G> {
		GreedyCappedLayeredGraphIndex {
			_phantom: self._phantom,
			data: data,
			graphs: self.graphs,
			local_layer_ids: self.local_layer_ids,
			global_layer_ids: self.global_layer_ids,
			distance: dist,
			max_frontier_size: self.max_frontier_size,
			higher_level_max_heap_size: self.higher_level_max_heap_size,
			entry_points: self.entry_points,
		}
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


