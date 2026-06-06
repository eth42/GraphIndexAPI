use ndarray::{Array1,Array2,ArrayView2};
use std::{pin::Pin, marker::PhantomData};
use futures::{prelude::*, executor::block_on};

use crate::data::{MatrixDataSource, AsyncMatrixDataSource};
use crate::types::{Number, StaticNumber};


pub struct H5PyDataset<T: Number> {
	_phantom: PhantomData<T>,
	file: String,
	dataset: String,
	n_rows: usize,
	n_cols: usize
}
impl<T: Number> H5PyDataset<T> {
	pub fn new(file: &str, dataset: &str) -> Self {
		let result: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|py| {
			let locals = pyo3::types::PyDict::new(py);
			locals.set_item("h5py", py.import("h5py")?)?;
			locals.set_item("data", py.eval(
				format!("h5py.File(\"{:}\")[\"{:}\"]", file, dataset).as_str(),
				None,
				Some(&locals)
			)?)?;
			let (n_rows, n_cols): (usize, usize) = py.eval(
				"data.shape",
				None,
				Some(&locals)
			)?.extract()?;
			Ok((n_rows, n_cols))
		});
		let (n_rows, n_cols) = result.unwrap();
		Self{
			_phantom: PhantomData,
			file: file.to_string(),
			dataset: dataset.to_string(),
			n_rows: n_rows,
			n_cols: n_cols
		}
	}
}
impl<T: Number> MatrixDataSource<T> for H5PyDataset<T> {
	const SUPPORTS_ROW_VIEW: bool = false;
	const SUPPORTS_ROW_SLICE_VIEW: bool = false;
	fn n_rows(&self) -> usize { self.n_rows }
	fn n_cols(&self) -> usize { self.n_cols }
	fn get_row(&self, i_row: usize) -> Array1<T> {
		let row: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|py| {
			let locals = pyo3::types::PyDict::new(py);
			locals.set_item("h5py", py.import("h5py")?)?;
			locals.set_item("np", py.import("numpy")?)?;
			locals.set_item("data", py.eval(
				format!("h5py.File(\"{:}\")[\"{:}\"]", self.file.as_str(), self.dataset.as_str()).as_str(),
				None,
				Some(&locals)
			)?)?;
			locals.set_item("i", i_row)?;
			let row_obj = py.eval(
				format!("data[i].astype(np.{:})", T::numpy_name()).as_str(),
				None,
				Some(&locals)
			)?;
			let row: &numpy::PyArray1<T> = row_obj.downcast()?;
			Ok(row.to_owned_array())
		});
		row.unwrap()
	}
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> {
		let row: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|py| {
			let locals = pyo3::types::PyDict::new(py);
			locals.set_item("h5py", py.import("h5py")?)?;
			locals.set_item("np", py.import("numpy")?)?;
			locals.set_item("data", py.eval(
				format!("h5py.File(\"{:}\")[\"{:}\"]", self.file.as_str(), self.dataset.as_str()).as_str(),
				None,
				Some(&locals)
			)?)?;
			locals.set_item("idx", i_rows)?;
			let row_obj = py.eval(
				format!("data[np.sort(idx)].astype(np.{:})", T::numpy_name()).as_str(),
				None,
				Some(&locals)
			)?;
			let row: &numpy::PyArray2<T> = row_obj.downcast()?;
			Ok(row.to_owned_array())
		});
		row.unwrap()
	}
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		let row: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|py| {
			let locals = pyo3::types::PyDict::new(py);
			locals.set_item("h5py", py.import("h5py")?)?;
			locals.set_item("np", py.import("numpy")?)?;
			locals.set_item("data", py.eval(
				format!("h5py.File(\"{:}\")[\"{:}\"]", self.file.as_str(), self.dataset.as_str()).as_str(),
				None,
				Some(&locals)
			)?)?;
			locals.set_item("start", i_row_from)?;
			locals.set_item("end", i_row_to)?;
			let row_obj = py.eval(
				format!("data[start:end].astype(np.{:})", T::numpy_name()).as_str(),
				None,
				Some(&locals)
			)?;
			let row: &numpy::PyArray2<T> = row_obj.downcast()?;
			Ok(row.to_owned_array())
		});
		row.unwrap()
	}
	fn get_row_view(&self, _i_row: usize) -> &[T] {
		panic!("Row view not supported for H5PyDataset");
	}
	fn get_row_slice_view(&self, _i_row_from: usize, _i_row_to: usize) -> ArrayView2<T> {
		panic!("Row slice view not supported for H5PyDataset");
	}
}

pub struct CachingH5PyReader<T: StaticNumber> {
	_phantom: PhantomData<T>,
	has_active_query: bool,
	query_is_range: bool,
	file_name: String,
	dataset_name: String,
	dataset: H5PyDataset<T>,
	cache_future: Option<Pin<Box<dyn Future<Output=Array2<T>>>>>
}
impl<T: StaticNumber> CachingH5PyReader<T> {
	pub fn new(file_name: String, dataset_name: String) -> Self {
		let dataset = H5PyDataset::<T>::new(file_name.as_str(), dataset_name.as_str());
		Self {
			_phantom: PhantomData,
			has_active_query: false,
			query_is_range: false,
			file_name: file_name,
			dataset_name: dataset_name,
			dataset: dataset,
			cache_future: None
		}
	}
	async fn load_rows(file_name: String, dataset_name: String, idx: Vec<usize>) -> Array2<T> {
		let data = H5PyDataset::<T>::new(file_name.as_str(), dataset_name.as_str());
		data.get_rows(&idx)
	}
	async fn load_rows_slice(file_name: String, dataset_name: String, start: usize, end: usize) -> Array2<T> {
		let data = H5PyDataset::<T>::new(file_name.as_str(), dataset_name.as_str());
		data.get_rows_slice(start, end)
	}
}
impl<T: StaticNumber> MatrixDataSource<T> for CachingH5PyReader<T> {
	const SUPPORTS_ROW_VIEW: bool = H5PyDataset::<T>::SUPPORTS_ROW_VIEW;
	const SUPPORTS_ROW_SLICE_VIEW: bool = H5PyDataset::<T>::SUPPORTS_ROW_SLICE_VIEW;
	fn n_rows(&self) -> usize {
		<H5PyDataset<T> as MatrixDataSource<T>>::n_rows(&self.dataset)
	}
	fn n_cols(&self) -> usize {
		<H5PyDataset<T> as MatrixDataSource<T>>::n_cols(&self.dataset)
	}
	fn get_row(&self, i_row: usize) -> Array1<T> {
		self.dataset.get_row(i_row)
	}
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> {
		self.dataset.get_rows(i_rows)
	}
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		self.dataset.get_rows_slice(i_row_from, i_row_to)
	}
	fn get_row_view(&self, i_row: usize) -> &[T] {
		self.dataset.get_row_view(i_row)
	}
	fn get_row_slice_view(&self, i_row_from: usize, i_row_to: usize) -> ArrayView2<T> {
		self.dataset.get_row_slice_view(i_row_from, i_row_to)
	}
}
impl<T: StaticNumber> AsyncMatrixDataSource<T> for CachingH5PyReader<T> {
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

