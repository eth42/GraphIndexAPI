use std::{iter::once, mem::transmute};

use ndarray::{ArrayView1, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Slice, s};

pub enum AbsRow<'a,T> {
	Owned(Array1<T>),
	Viewed(&'a [T]),
}
impl<'a,T> AbsRow<'a,T> {
	pub fn as_slice(&self) -> &[T] {
		match self {
			AbsRow::Owned(arr) => arr.as_slice().unwrap(),
			AbsRow::Viewed(slice) => slice,
		}
	}
	pub fn as_view(&'a self) -> ArrayView1<'a,T> {
		match self {
			AbsRow::Owned(arr) => arr.view(),
			AbsRow::Viewed(slice) => unsafe{ArrayView1::from_shape_ptr((slice.len(),), slice.as_ptr())},
		}
	}
}

pub trait MatrixDataSource<T> {
	const SUPPORTS_ROW_VIEW: bool;
	const SUPPORTS_ROW_SLICE_VIEW: bool;
	fn n_rows(&self) -> usize;
	fn n_cols(&self) -> usize;
	fn get_abs_row(&self, i_row: usize) -> AbsRow<T> {
		if Self::SUPPORTS_ROW_VIEW {
			AbsRow::Viewed(self.get_row_view(i_row))
		} else {
			AbsRow::Owned(self.get_row(i_row))
		}
	}
	fn get_row(&self, i_row: usize) -> Array1<T>;
	fn get_row_view(&self, i_row: usize) -> &[T];
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T>;
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T>;
	fn get_row_slice_view(&self, i_row_from: usize, i_row_to: usize) -> ArrayView2<T>;
}
pub trait AsyncMatrixDataSource<T>: MatrixDataSource<T> {
	fn prepare_rows(&mut self, i_rows: Vec<usize>) -> Result<(), ()>;
	fn prepare_rows_slice(&mut self, i_row_from: usize, i_row_to: usize) -> Result<(), ()>;
	fn get_cached(&mut self) -> Option<Array2<T>>;
}


impl<T, M: MatrixDataSource<T>> MatrixDataSource<T> for &M {
	const SUPPORTS_ROW_VIEW: bool = M::SUPPORTS_ROW_VIEW;
	const SUPPORTS_ROW_SLICE_VIEW: bool = M::SUPPORTS_ROW_SLICE_VIEW;
	#[inline(always)]
	fn n_rows(&self) -> usize { M::n_rows(&self) }
	#[inline(always)]
	fn n_cols(&self) -> usize { M::n_cols(&self) }
	#[inline(always)]
	fn get_row(&self, i_row: usize) -> Array1<T> { M::get_row(&self, i_row) }
	#[inline(always)]
	fn get_row_view(&self, i_row: usize) -> &[T] { M::get_row_view(&self, i_row) }
	#[inline(always)]
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> { M::get_rows(&self, i_rows) }
	#[inline(always)]
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> { M::get_rows_slice(&self, i_row_from, i_row_to) }
	#[inline(always)]
	fn get_row_slice_view(&self, i_row_from: usize, i_row_to: usize) -> ArrayView2<T> { M::get_row_slice_view(&self, i_row_from, i_row_to) }
}
impl<T: Copy+Clone, D: Data<Elem=T>> MatrixDataSource<T> for ArrayBase<D, Ix2> {
	const SUPPORTS_ROW_VIEW: bool = true;
	const SUPPORTS_ROW_SLICE_VIEW: bool = true;
	#[inline(always)]
	fn n_rows(&self) -> usize {
		self.shape()[0]
	}
	#[inline(always)]
	fn n_cols(&self) -> usize {
		self.shape()[1]
	}
	#[inline(always)]
	fn get_row(&self, i_row: usize) -> Array1<T> {
		self.row(i_row).into_owned()
	}
	#[inline(always)]
	fn get_row_view(&self, i_row: usize) -> &[T] {
		self.as_slice().unwrap().split_at(i_row*self.n_cols()).1.split_at(self.n_cols()).0
	}
	#[inline(always)]
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> {
		Array2::from_shape_fn(
			(i_rows.len(), self.n_cols()),
			|(i,j)| self[[i_rows[i], j]]
		)
	}
	#[inline(always)]
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		self.slice_axis(Axis(0), Slice::from(i_row_from..i_row_to.min(self.n_rows()))).to_owned()
	}
	#[inline(always)]
	fn get_row_slice_view(&self, i_row_from: usize, i_row_to: usize) -> ArrayView2<T> {
		self.slice_axis(Axis(0), Slice::from(i_row_from..i_row_to.min(self.n_rows())))
	}
}


pub trait TransmuteInto<T> { fn transmute(self) -> T; }
impl<A: Clone+TransmuteInto<B>,B> TransmuteInto<B> for &A {
	fn transmute(self) -> B {
		self.clone().transmute()
	}
}
macro_rules! make_transmutes {
	($t_float: ty, $t_uint: ty) => {
		impl TransmuteInto<$t_uint> for $t_float {
			#[inline(always)]
			fn transmute(self) -> $t_uint {
				unsafe{transmute(self)}
			}
		}
		impl TransmuteInto<$t_float> for $t_uint {
			#[inline(always)]
			fn transmute(self) -> $t_float {
				unsafe{transmute(self)}
			}
		}
	}
}
make_transmutes!(half::f16, i16);
make_transmutes!(half::f16, u16);
make_transmutes!(f32, i32);
make_transmutes!(f32, u32);
make_transmutes!(f64, i64);
make_transmutes!(f64, u64);

/// A storage structure for row-major sparse matrices where columns
/// and column values are stored interleaved (alternating starting
/// with the index).
/// For this to work, the stored values must have an equivalently
/// sized uint type.
pub struct InterleavedSparseMatrix<T: Copy+Clone> {
	row_limits: Array1<usize>,
	interleaved_data: Array1<T>,
	n_rows: usize,
	n_cols: usize,
}
impl<T: Copy+Clone> InterleavedSparseMatrix<T> {
	pub fn from_csr<
		R1: Num+std::cmp::PartialOrd+NumCast+Copy+Clone+TransmuteInto<T>,
		R2: Num+std::cmp::PartialOrd+NumCast+Copy+Clone,
		D1: Data<Elem=T>,
		D2: Data<Elem=R1>,
		D3: Data<Elem=R2>,
	>(data: ArrayBase<D1,Ix1>, indices: ArrayBase<D2,Ix1>, indptr: ArrayBase<D3,Ix1>, n_cols: Option<usize>) -> Self {
		let n_cols = n_cols.unwrap_or(
			indices.iter()
			.reduce(|a,b| if a.partial_cmp(b).unwrap().is_ge() {a} else {b}).unwrap()
			.to_usize().unwrap() + 1
		);
		let n_rows = indptr.len()-1;
		let interleaved_data = Array1::from_iter(
			indices.into_iter()
			.zip(data.into_iter())
			.map(|(&a,&b)| once(a.transmute()).chain(once(b)))
			.flatten()
		);
		Self {
			row_limits: indptr.mapv(|v| v.to_usize().unwrap() * 2),
			interleaved_data: interleaved_data,
			n_rows,
			n_cols,
		}
	}
	pub fn from_lol<R: Num+std::cmp::PartialOrd+NumCast+Copy+Clone+TransmuteInto<T>>(data: Vec<Vec<T>>, rows: Vec<Vec<R>>, n_cols: Option<usize>) -> Self {
		let n_cols: usize = n_cols.unwrap_or(
			rows.iter()
			.map(|row| {
				row.iter()
				.reduce(|a,b| if a.partial_cmp(b).unwrap().is_ge() {a} else {b})
				.cloned()
				.unwrap_or(R::zero())
			})
			.reduce(|a,b| if a.partial_cmp(&b).unwrap().is_ge() {a} else {b})
			.unwrap_or(R::zero())
			.to_usize().unwrap() + 1
		);
		let n_rows = data.len();
		let mut row_limits = Array1::zeros((n_rows+1,));
		for i in 0..n_rows {
			row_limits[i+1] = row_limits[i] + data[i].len() * 2;
		}
		let interleaved_data = Array1::from_iter(
			rows.into_iter()
			.zip(data.into_iter())
			.map(|(irow, drow)| {
				irow.into_iter()
				.zip(drow.into_iter())
				.map(|(i,d)| once(i.transmute()).chain(once(d)))
				.flatten()
			})
			.flatten()
		);
		Self {
			row_limits,
			interleaved_data: interleaved_data,
			n_rows,
			n_cols,
		}
	}
}
impl<T: Copy+Clone> MatrixDataSource<T> for InterleavedSparseMatrix<T> {
	const SUPPORTS_ROW_VIEW: bool = true;
	const SUPPORTS_ROW_SLICE_VIEW: bool = false;
	fn n_rows(&self) -> usize { self.n_rows }
	fn n_cols(&self) -> usize { self.n_cols }
	fn get_row(&self, i_row: usize) -> Array1<T> {
		let start = self.row_limits[i_row];
		let end = self.row_limits[i_row+1];
		self.interleaved_data.slice(s![start..end]).to_owned()
	}
	fn get_row_view(&self, i_row: usize) -> &[T] {
		let start = self.row_limits[i_row];
		let end = self.row_limits[i_row+1];
		self.interleaved_data.slice(s![start..end]).to_slice().unwrap()
	}
	fn get_rows(&self, _i_rows: &Vec<usize>) -> Array2<T> {
		panic!("Not implemented for sparse matrices")
	}
	fn get_rows_slice(&self, _i_row_from: usize, _i_row_to: usize) -> Array2<T> {
		panic!("Not implemented for sparse matrices")
	}
	fn get_row_slice_view(&self, _i_row_from: usize, _i_row_to: usize) -> ArrayView2<T> {
		panic!("Not implemented for sparse matrices")
	}
}

#[test]
fn interleaved_sparse_matrix_test() {
	let n_rows = 100usize;
	let n_cols = 100usize;
	let n_vals_per_col = 20;
	/* Create unique sorted column indices for each row */
	let vec_indices: Vec<Vec<i32>> = (0..n_rows).into_iter().map(|_| {
		let mut indices: Vec<i32> = Vec::with_capacity(n_vals_per_col);
		for val in (0..n_vals_per_col).map(|_| (rand::random::<usize>() % n_cols) as i32) {
			if !indices.contains(&val) { indices.push(val); }
		}
		indices.sort_unstable();
		indices
	}).collect();
	/* Collect CSR arrays */
	let indices = Array1::from_iter(vec_indices.iter().map(|v| v.iter().cloned()).flatten());
	let data = Array1::from_iter(indices.iter().map(|_| rand::random::<f32>()));
	let mut indptr = Array1::from_iter(std::iter::once(0i32).chain(vec_indices.iter().map(|v| v.len() as i32)));
	for i in 1..indptr.len() { indptr[i] += indptr[i-1]; }
	assert!(indptr[0] == 0i32);
	assert!(indptr[indptr.len()-1] == data.len() as i32);
	assert!(indptr.len() == n_rows+1);
	/* Create dense matrix */
	let mut dense_mat = Array2::zeros((n_rows,n_cols));
	indptr.iter().zip(indptr.iter().skip(1)).enumerate()
	.for_each(|(i_row, (&start, &end))| {
		assert!(i_row < dense_mat.len_of(Axis(0)));
		let (start, end) = (start as usize, end as usize);
		indices.slice_axis(Axis(0), Slice::from(start..end)).iter()
		.zip(data.slice_axis(Axis(0), Slice::from(start..end)).iter())
		.for_each(|(&i,&v)| dense_mat[[i_row, i as usize]] = v);
	});
	/* Create sparse matrix */
	let sparse_mat = InterleavedSparseMatrix::from_csr(data, indices, indptr, None);
	assert!(n_rows == sparse_mat.n_rows);
	assert!(n_cols >= sparse_mat.n_cols);
	/* Compare products */
	use crate::measures::{Distance, SparseNegDotProduct, NegDotProduct, SparseSquaredEuclideanDistance, SquaredEuclideanDistance};
	let sparse_dot = SparseNegDotProduct::<f32,i32>::new();
	let dense_dot = NegDotProduct::<f32>::new();
	let sparse_sqeuc = SparseSquaredEuclideanDistance::<f32,i32>::new();
	let dense_sqeuc = SquaredEuclideanDistance::<f32>::new();
	for i in 0..n_rows {
		for j in 0..i {
			let sparse_val = sparse_dot.dist_slice(sparse_mat.get_row_view(i), sparse_mat.get_row_view(j));
			let dense_val = dense_dot.dist_slice(dense_mat.get_row_view(i), dense_mat.get_row_view(j));
			assert!((sparse_val-dense_val).abs() < 1e-5);
			let sparse_val = sparse_sqeuc.dist_slice(sparse_mat.get_row_view(i), sparse_mat.get_row_view(j));
			let dense_val = dense_sqeuc.dist_slice(dense_mat.get_row_view(i), dense_mat.get_row_view(j));
			assert!((sparse_val-dense_val).abs() < 1e-5);
		}
	}
}


#[cfg(feature="hdf5")]
mod hdf5_defs {
	use std::marker::PhantomData;
	use std::pin::Pin;
	use futures::{prelude::*, executor::block_on};
	use rayon::iter::{ParallelBridge, ParallelIterator};
	use ndarray::s;
	use crate::types::{SyncFloat, StaticSyncFloat};

	impl<T: SyncFloat> MatrixDataSource<T> for hdf5::Dataset {
		const SUPPORTS_ROW_VIEW: bool = false;
		const SUPPORTS_ROW_SLICE_VIEW: bool = false;
		#[inline(always)]
		fn n_rows(&self) -> usize {
			self.shape()[0]
		}
		#[inline(always)]
		fn n_cols(&self) -> usize {
			self.shape()[1]
		}
		#[inline(always)]
		fn get_row(&self, i_row: usize) -> Array1<T> {
			self.read_slice_1d(s![i_row, ..]).unwrap()
		}
		#[inline(always)]
		fn get_row_view(&self, _i_row: usize) -> &[T] {
			panic!("Not implemented");
		}
		#[inline(always)]
		fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> {
			let n_rows = i_rows.len();
			let n_cols = <hdf5::Dataset as MatrixDataSource<T>>::n_cols(&self);
			let mut ret: Array2<T> = Array2::from_elem([n_rows, n_cols], T::zero());
			ret.axis_iter_mut(Axis(0))
			.zip(i_rows.into_iter())
			.par_bridge()
			.for_each(|(mut target, i_row)| {
				target.assign(&self.get_row(*i_row));
			});
			ret
			// let ret_vec: Vec<Array1<T>> = i_rows.into_iter()
			// .map(|i_row| self.get_row(i_row))
			// .collect();
			// let shape: (usize, usize) = (n_rows, <hdf5::Dataset as MatrixDataSource<T>>::n_cols(&self));
			// Array2::from_shape_fn(
			// 	shape,
			// 	|(i,j)| unsafe { *ret_vec.get_unchecked(i).uget(j) }
			// )
		}
	
		fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
			let n_rows = <hdf5::Dataset as MatrixDataSource<T>>::n_rows(&self);
			self.read_slice_2d(s![i_row_from..i_row_to.min(n_rows),..]).unwrap()
		}
		#[inline(always)]
		fn get_row_slice_view(&self, i_row_from: usize, i_row_to: usize) -> ArrayView2<T> {
			panic!("Not implemented");
		}
	}

	pub struct CachingH5Reader<T: StaticSyncFloat> {
		_phantom: PhantomData<T>,
		has_active_query: bool,
		query_is_range: bool,
		file_name: String,
		dataset_name: String,
		dataset: hdf5::Dataset,
		cache_future: Option<Pin<Box<dyn Future<Output=Array2<T>>>>>
	}
	impl<T: StaticSyncFloat> CachingH5Reader<T> {
		#[allow(unused)]
		pub fn new(file_name: String, dataset_name: String) -> Self {
			let dataset = read_h5_dataset(file_name.as_str(), dataset_name.as_str());
			Self {
				_phantom: PhantomData,
				has_active_query: false,
				query_is_range: false,
				file_name: file_name,
				dataset_name: dataset_name,
				dataset: dataset.unwrap(),
				cache_future: None
			}
		}
		async fn load_rows(file_name: String, dataset_name: String, idx: Vec<usize>) -> Array2<T> {
			let data = read_h5_dataset(file_name.as_str(), dataset_name.as_str());
			data.unwrap().get_rows(&idx)
		}
		async fn load_rows_slice(file_name: String, dataset_name: String, start: usize, end: usize) -> Array2<T> {
			let data = read_h5_dataset(file_name.as_str(), dataset_name.as_str());
			data.unwrap().get_rows_slice(start, end)
		}
	}
	impl<T: StaticSyncFloat> MatrixDataSource<T> for CachingH5Reader<T> {
		const SUPPORTS_ROW_VIEW: bool = false;
		const SUPPORTS_ROW_SLICE_VIEW: bool = false;
		#[inline(always)]
		fn n_rows(&self) -> usize {
			<hdf5::Dataset as MatrixDataSource<T>>::n_rows(&self.dataset)
		}
		#[inline(always)]
		fn n_cols(&self) -> usize {
			<hdf5::Dataset as MatrixDataSource<T>>::n_cols(&self.dataset)
		}
		#[inline(always)]
		fn get_row(&self, i_row: usize) -> Array1<T> {
			self.dataset.get_row(i_row)
		}
		#[inline(always)]
		fn get_row_view(&self, i_row: usize) -> &[T] {
			self.dataset.get_row_view(i_row)
		}
		#[inline(always)]
		fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> {
			self.dataset.get_rows(i_rows)
		}
		#[inline(always)]
		fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
			self.dataset.get_rows_slice(i_row_from, i_row_to)
		}
		#[inline(always)]
		fn get_row_slice_view(&self, i_row_from: usize, i_row_to: usize) -> ArrayView2<T> {
			panic!("Not implemented");
		}
	}
	impl<T: StaticSyncFloat> AsyncMatrixDataSource<T> for CachingH5Reader<T> {
		fn prepare_rows(&mut self, idx: Vec<usize>) -> Result<(), ()> {
			if self.has_active_query {
				Err(())
			} else {
				self.query_is_range = false;
				self.has_active_query = true;
				self.cache_future = Some(Box::pin(Self::load_rows(self.file_name.clone(), self.dataset_name.clone(), idx)));
				Ok(())
			}
		}
		fn prepare_rows_slice(&mut self, start: usize, end: usize) -> Result<(), ()> {
			if self.has_active_query {
				Err(())
			} else {
				self.query_is_range = true;
				self.has_active_query = true;
				self.cache_future = Some(Box::pin(Self::load_rows_slice(self.file_name.clone(), self.dataset_name.clone(), start, end)));
				Ok(())
			}
		}
		fn get_cached(&mut self) -> Option<Array2<T>> {
			if self.cache_future.is_none() || !self.has_active_query {
				None
			} else {
				let future_arr = unsafe{self.cache_future.take().unwrap_unchecked()};
				let arr = block_on(future_arr);
				self.has_active_query = false;
				Some(arr)
			}
		}
	}

	pub fn read_h5_dataset(file: &str, dataset: &str) -> Result<hdf5::Dataset, hdf5::Error> {
		let file = hdf5::File::open(file)?;
		file.dataset(dataset)
	}
	#[allow(unused)]
	pub fn store_h5_dataset<T: hdf5::H5Type>(file: &str, dataset: &str, data: &Array2<T>) -> Result<(), hdf5::Error>{
		let file = hdf5::File::create(file)?;
		let dataset_builder = file.new_dataset_builder();
		dataset_builder.with_data(data).create(dataset)?;
		Ok(())
	}
}
#[cfg(feature="hdf5")]
pub use hdf5_defs::*;
use num::{Num, NumCast};
