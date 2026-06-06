use std::iter::Iterator;
use ndarray::{Array1, ArrayBase, Ix1, ViewRepr, OwnedRepr, ArrayViewMut1, ArrayView1};

use crate::bits::Bits;

// Trait to unify bit vector types
pub trait BitVector {
	type BitIterator<'a>: Iterator<Item=bool> where Self: 'a;

	fn size(&self) -> usize;
	fn iter_bits<'a>(&'a self) -> Self::BitIterator<'a>;
	fn get_bit(&self, i: usize) -> Option<bool> {
		if i < self.size() { Some(self.get_bit_unchecked(i)) } else { None }
	}
	fn get_bit_unchecked(&self, i: usize) -> bool;
	fn count_bits(&self) -> usize;
	fn count_bits_range(&self, lo: usize, hi: usize) -> Option<usize> {
		if lo >= hi || hi > self.size() { None } else { Some(self.count_bits_range_unchecked(lo, hi)) }
	}
	fn count_bits_range_unchecked(&self, lo: usize, hi: usize) -> usize;

	fn hamming_dist<V: BitVector>(&self, other: &V) -> usize {
		self.iter_bits().zip(other.iter_bits()).filter(|(a,b)| a != b).count()
	}
	fn dot_prod<V: BitVector>(&self, other: &V) -> usize {
		self.iter_bits().zip(other.iter_bits()).filter(|(a,b)| a & b).count()
	}
	fn hamming_dist_same(&self, other: &Self) -> usize;
	fn dot_prod_same(&self, other: &Self) -> usize;

	fn or<V: BitVector, R: BitVectorMut+BitVectorOwned>(&self, other: &V) -> R {
		let mut ret = R::zeros(self.size());
		self.iter_bits()
		.zip(other.iter_bits())
		.enumerate()
		.for_each(|(i_bit, (a, b))| ret.set_bit(i_bit, a | b));
		ret
	}
	fn and<V: BitVector, R: BitVectorMut+BitVectorOwned>(&self, other: &V) -> R {
		let mut ret = R::zeros(self.size());
		self.iter_bits()
		.zip(other.iter_bits())
		.enumerate()
		.for_each(|(i_bit, (a, b))| ret.set_bit(i_bit, a & b));
		ret
	}
	fn xor<V: BitVector, R: BitVectorMut+BitVectorOwned>(&self, other: &V) -> R {
		let mut ret = R::zeros(self.size());
		self.iter_bits()
		.zip(other.iter_bits())
		.enumerate()
		.for_each(|(i_bit, (a, b))| ret.set_bit(i_bit, a ^ b));
		ret
	}
	fn not<R: BitVectorMut+BitVectorOwned>(&self) -> R {
		let mut ret = R::zeros(self.size());
		self.iter_bits()
		.enumerate()
		.for_each(|(i_bit, a)| ret.set_bit(i_bit, !a));
		ret
	}
}
pub trait BitVectorMut: BitVector {
	fn set_bit(&mut self, i: usize, v: bool) {
		if i < self.size() { self.set_bit_unchecked(i, v); }
	}
	fn set_true(&mut self, i: usize) {
		self.set_bit(i, true);
	}
	fn set_false(&mut self, i: usize) {
		self.set_bit(i, false);
	}
	fn flip(&mut self, i: usize) {
		if i < self.size() { self.flip_unchecked(i); }
	}
	fn set_bit_unchecked(&mut self, i: usize, v: bool);
	fn set_true_unchecked(&mut self, i: usize) {
		self.set_bit_unchecked(i, true);
	}
	fn set_false_unchecked(&mut self, i: usize) {
		self.set_bit_unchecked(i, false);
	}
	fn flip_unchecked(&mut self, i: usize) {
		self.set_bit_unchecked(i, !self.get_bit_unchecked(i));
	}
}
pub trait BitVectorOwned: BitVector {
	fn zeros(capacity: usize) -> Self;
	fn ones(capacity: usize) -> Self;
}
pub trait BitVectorMutOwned: BitVectorMut+BitVectorOwned {
	fn not_same(&self) -> Self;
	fn or_same(&self, other: &Self) -> Self;
	fn and_same(&self, other: &Self) -> Self;
	fn xor_same(&self, other: &Self) -> Self;
}
pub struct BitVectorWrapper<V: BitVector> {
	vec: V
}
impl<V: BitVector> BitVectorWrapper<V> {
	#[allow(unused)]
	pub fn unwrap(self) -> V {
		self.vec
	}
}
impl<V: BitVector> BitVector for BitVectorWrapper<V> {
	type BitIterator<'a> = <V as BitVector>::BitIterator<'a> where Self: 'a;
	fn size(&self) -> usize { self.vec.size() }
	fn iter_bits<'a>(&'a self) -> Self::BitIterator<'a> { self.vec.iter_bits() }
	fn get_bit_unchecked(&self, i: usize) -> bool { self.vec.get_bit_unchecked(i) }
	fn count_bits(&self) -> usize { self.vec.count_bits() }
	fn count_bits_range_unchecked(&self, lo: usize, hi: usize) -> usize { self.vec.count_bits_range_unchecked(lo, hi) }
	fn hamming_dist_same(&self, other: &Self) -> usize { self.vec.hamming_dist_same(&other.vec) }
	fn dot_prod_same(&self, other: &Self) -> usize { self.vec.dot_prod_same(&other.vec) }
}
impl<V: BitVectorMut> BitVectorMut for BitVectorWrapper<V> {
	fn set_bit_unchecked(&mut self, i: usize, v: bool) { self.vec.set_bit_unchecked(i,v) }
}
impl<V: BitVectorOwned> BitVectorOwned for BitVectorWrapper<V> {
	fn zeros(capacity: usize) -> Self {
		BitVectorWrapper{vec: <V as BitVectorOwned>::zeros(capacity) }
	}
	fn ones(capacity: usize) -> Self {
		BitVectorWrapper{vec: <V as BitVectorOwned>::ones(capacity) }
	}
}
impl<V: BitVectorMutOwned> BitVectorMutOwned for BitVectorWrapper<V> {
	fn not_same(&self) -> Self {
		BitVectorWrapper{vec: <V as BitVectorMutOwned>::not_same(&self.vec) }
	}
	fn or_same(&self, other: &Self) -> Self {
		BitVectorWrapper{vec: <V as BitVectorMutOwned>::or_same(&self.vec, &other.vec) }
	}
	fn and_same(&self, other: &Self) -> Self {
		BitVectorWrapper{vec: <V as BitVectorMutOwned>::and_same(&self.vec, &other.vec) }
	}
	fn xor_same(&self, other: &Self) -> Self {
		BitVectorWrapper{vec: <V as BitVectorMutOwned>::xor_same(&self.vec, &other.vec) }
	}
}
impl<V: BitVectorOwned+BitVectorMut, I: ExactSizeIterator<Item=bool>> From<I> for BitVectorWrapper<V> {
	fn from(iter: I) -> Self {
		let mut s: BitVectorWrapper<V> = BitVectorWrapper{vec: <V as BitVectorOwned>::zeros(iter.len())};
		iter.enumerate().for_each(|(i,bit)| s.set_bit(i, bit));
		s
	}
}

// Wrapping Iterator to transfer arbitrary Bits types to booleans
pub struct BitsToBoolIterator<'a, B: Bits, It: Iterator<Item=&'a B>> {
	wrapped_iterator: It,
	bit_offset: usize,
	curr_val: Option<&'a B>
}
impl<'a, B: Bits, It: Iterator<Item=&'a B>> BitsToBoolIterator<'a, B, It> {
	pub fn new(iter: It) -> Self {
		Self {
			wrapped_iterator: iter,
			bit_offset: 0,
			curr_val: None
		}
	}
}
impl<'a, B: Bits, It: Iterator<Item=&'a B>> Iterator for BitsToBoolIterator<'a, B, It> {
	type Item=bool;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if self.bit_offset == 0 {
			self.curr_val = self.wrapped_iterator.next();
		}
		if self.curr_val.is_none() {
			None
		} else {
			unsafe {
				let bit = self.curr_val.unwrap_unchecked().get_bit(self.bit_offset);
				self.bit_offset += 1;
				if self.bit_offset == B::size() {
					self.bit_offset = 0;
				}
				bit
			}
		}
	}
}


impl<'b, B: Bits> BitVector for &'b [B] {
	type BitIterator<'a> = BitsToBoolIterator<'a, B, std::slice::Iter<'a, B>> where B: 'a, 'b: 'a;

	#[inline(always)]
	fn size(&self) -> usize {
		self.len() * B::size()
	}
	#[inline(always)]
	fn iter_bits<'a>(&'a self) -> Self::BitIterator<'a> {
		BitsToBoolIterator::new(self.iter())
	}
	#[inline(always)]
	fn get_bit_unchecked(&self, i: usize) -> bool {
		let entry = i / B::size();
		let bit_index = i % B::size();
		unsafe { self.get_unchecked(entry).get_bit_unchecked(bit_index) }
	}
	#[inline(always)]
	fn hamming_dist_same(&self, other: &Self) -> usize {
		self.iter()
		.zip(other.iter())
		.map(|(a,b)| a.hamming_dist(b))
		.sum()
	}
	#[inline(always)]
	fn dot_prod_same(&self, other: &Self) -> usize {
		self.iter()
		.zip(other.iter())
		.map(|(a,b)| a.dot_prod(b))
		.sum()
	}

	#[inline(always)]
	fn count_bits(&self) -> usize {
		self.iter().map(|bits| bits.count_bits()).sum()
	}
	#[inline(always)]
	fn count_bits_range_unchecked(&self, lo: usize, hi: usize) -> usize {
		let first_item = lo / B::size();
		let last_item = (hi-1) / B::size();
		let first_bit_offset = lo % B::size();
		let last_bit_offset = (hi-1) % B::size();
		unsafe { 
			if first_item == last_item {
				self.get_unchecked(first_item).count_bits_range_unchecked(first_bit_offset, last_bit_offset+1)
			} else {
				let mut cnt = 0;
				if first_bit_offset == 0 {
					cnt += self.get_unchecked(first_item).count_bits();
				} else {
					cnt += self.get_unchecked(first_item).count_bits_range_unchecked(first_bit_offset, B::size());
				}
				if last_bit_offset+1 == B::size() {
					cnt += self.get_unchecked(last_item).count_bits();
				} else {
					cnt += self.get_unchecked(last_item).count_bits_range_unchecked(0, last_bit_offset+1);
				}
				cnt += (first_item+1..last_item).map(|i| self.get_unchecked(i).count_bits()).sum::<usize>();
				cnt
			}
		}
	}
}
impl<B: Bits> BitVector for Vec<B> {
	type BitIterator<'a> = BitsToBoolIterator<'a, B, std::slice::Iter<'a, B>> where B: 'a;

	#[inline(always)]
	fn size(&self) -> usize {
		self.len() * B::size()
	}
	#[inline(always)]
	fn iter_bits<'a>(&'a self) -> Self::BitIterator<'a> {
		BitsToBoolIterator::new(<[B]>::iter(&self))
	}
	#[inline(always)]
	fn get_bit_unchecked(&self, i: usize) -> bool {
		let entry = i / B::size();
		let bit_index = i % B::size();
		unsafe { self.get_unchecked(entry).get_bit_unchecked(bit_index) }
	}
	#[inline(always)]
	fn hamming_dist_same(&self, other: &Self) -> usize {
		<[B]>::iter(&self)
		.zip(<[B]>::iter(&other))
		.map(|(a,b)| a.hamming_dist(b))
		.sum()
	}
	#[inline(always)]
	fn dot_prod_same(&self, other: &Self) -> usize {
		<[B]>::iter(&self)
		.zip(<[B]>::iter(&other))
		.map(|(a,b)| a.dot_prod(b))
		.sum()
	}

	#[inline(always)]
	fn count_bits(&self) -> usize {
		self.iter().map(|bits| bits.count_bits()).sum()
	}
	#[inline(always)]
	fn count_bits_range_unchecked(&self, lo: usize, hi: usize) -> usize {
		let first_item = lo / B::size();
		let last_item = (hi-1) / B::size();
		let first_bit_offset = lo % B::size();
		let last_bit_offset = (hi-1) % B::size();
		unsafe { 
			if first_item == last_item {
				self.get_unchecked(first_item).count_bits_range_unchecked(first_bit_offset, last_bit_offset+1)
			} else {
				let mut cnt = 0;
				if first_bit_offset == 0 {
					cnt += self.get_unchecked(first_item).count_bits();
				} else {
					cnt += self.get_unchecked(first_item).count_bits_range_unchecked(first_bit_offset, B::size());
				}
				if last_bit_offset+1 == B::size() {
					cnt += self.get_unchecked(last_item).count_bits();
				} else {
					cnt += self.get_unchecked(last_item).count_bits_range_unchecked(0, last_bit_offset+1);
				}
				cnt += (first_item+1..last_item).map(|i| self.get_unchecked(i).count_bits()).sum::<usize>();
				cnt
			}
		}
	}
}
impl<B: Bits> BitVector for Array1<B> {
	type BitIterator<'a> = BitsToBoolIterator<'a, B, ndarray::iter::Iter<'a, B, Ix1>> where B: 'a;

	#[inline(always)]
	fn size(&self) -> usize {
		self.len() * B::size()
	}
	#[inline(always)]
	fn iter_bits<'a>(&'a self) -> Self::BitIterator<'a> {
		BitsToBoolIterator::new(<ArrayBase<OwnedRepr<B>,Ix1>>::iter(&self))
	}
	#[inline(always)]
	fn get_bit_unchecked(&self, i: usize) -> bool {
		let entry = i / B::size();
		let bit_index = i % B::size();
		unsafe { self.uget(entry).get_bit_unchecked(bit_index) }
	}
	#[inline(always)]
	fn hamming_dist_same(&self, other: &Self) -> usize {
		<ArrayBase<OwnedRepr<B>,Ix1>>::iter(&self)
		.zip(<ArrayBase<OwnedRepr<B>,Ix1>>::iter(&other))
		.map(|(a,b)| a.hamming_dist(b))
		.sum()
	}
	#[inline(always)]
	fn dot_prod_same(&self, other: &Self) -> usize {
		<ArrayBase<OwnedRepr<B>,Ix1>>::iter(&self)
		.zip(<ArrayBase<OwnedRepr<B>,Ix1>>::iter(&other))
		.map(|(a,b)| a.dot_prod(b))
		.sum()
	}

	#[inline(always)]
	fn count_bits(&self) -> usize {
		self.iter().map(|bits| bits.count_bits()).sum()
	}
	#[inline(always)]
	fn count_bits_range_unchecked(&self, lo: usize, hi: usize) -> usize {
		let first_item = lo / B::size();
		let last_item = (hi-1) / B::size();
		let first_bit_offset = lo % B::size();
		let last_bit_offset = (hi-1) % B::size();
		unsafe {
			if first_item == last_item {
				self.uget(first_item).count_bits_range_unchecked(first_bit_offset, last_bit_offset+1)
			} else {
				let mut cnt = 0;
				if first_bit_offset == 0 {
					cnt += self.uget(first_item).count_bits();
				} else {
					cnt += self.uget(first_item).count_bits_range_unchecked(first_bit_offset, B::size());
				}
				if last_bit_offset+1 == B::size() {
					cnt += self.uget(last_item).count_bits();
				} else {
					cnt += self.uget(last_item).count_bits_range_unchecked(0, last_bit_offset+1);
				}
				cnt += (first_item+1..last_item).map(|i| self.uget(i).count_bits()).sum::<usize>();
				cnt
			}
		}
	}
}
impl<'x, B: Bits> BitVector for ArrayView1<'x, B> {
	type BitIterator<'a> = BitsToBoolIterator<'a, B, ndarray::iter::Iter<'a, B, Ix1>> where B: 'a, 'x: 'a;

	#[inline(always)]
	fn size(&self) -> usize {
		self.len() * B::size()
	}
	#[inline(always)]
	fn iter_bits<'a>(&'a self) -> Self::BitIterator<'a> {
		BitsToBoolIterator::new(<ArrayBase<ViewRepr<&'x B>,Ix1>>::iter(&self))
	}
	#[inline(always)]
	fn get_bit_unchecked(&self, i: usize) -> bool {
		let entry = i / B::size();
		let bit_index = i % B::size();
		unsafe { self.uget(entry).get_bit_unchecked(bit_index) }
	}
	#[inline(always)]
	fn hamming_dist_same<'y>(&self, other: &ArrayView1<'y, B>) -> usize {
		<ArrayBase<ViewRepr<&'x B>,Ix1>>::iter(&self)
		.zip(<ArrayBase<ViewRepr<&'y B>,Ix1>>::iter(&other))
		.map(|(a,b)| a.hamming_dist(b))
		.sum()
	}
	#[inline(always)]
	fn dot_prod_same(&self, other: &Self) -> usize {
		<ArrayBase<ViewRepr<&'x B>,Ix1>>::iter(&self)
		.zip(<ArrayBase<ViewRepr<&'x B>,Ix1>>::iter(&other))
		.map(|(a,b)| a.dot_prod(b))
		.sum()
	}

	#[inline(always)]
	fn count_bits(&self) -> usize {
		self.iter().map(|bits| bits.count_bits()).sum()
	}
	#[inline(always)]
	fn count_bits_range_unchecked(&self, lo: usize, hi: usize) -> usize {
		let first_item = lo / B::size();
		let last_item = (hi-1) / B::size();
		let first_bit_offset = lo % B::size();
		let last_bit_offset = (hi-1) % B::size();
		unsafe {
			if first_item == last_item {
				self.uget(first_item).count_bits_range_unchecked(first_bit_offset, last_bit_offset+1)
			} else {
				let mut cnt = 0;
				if first_bit_offset == 0 {
					cnt += self.uget(first_item).count_bits();
				} else {
					cnt += self.uget(first_item).count_bits_range_unchecked(first_bit_offset, B::size());
				}
				if last_bit_offset+1 == B::size() {
					cnt += self.uget(last_item).count_bits();
				} else {
					cnt += self.uget(last_item).count_bits_range_unchecked(0, last_bit_offset+1);
				}
				cnt += (first_item+1..last_item).map(|i| self.uget(i).count_bits()).sum::<usize>();
				cnt
			}
		}
	}
}
impl<'x, B: Bits> BitVector for ArrayViewMut1<'x, B> {
	type BitIterator<'a> = BitsToBoolIterator<'a, B, ndarray::iter::Iter<'a, B, Ix1>> where B: 'a, 'x: 'a;

	#[inline(always)]
	fn size(&self) -> usize {
		self.len() * B::size()
	}
	#[inline(always)]
	fn iter_bits<'a>(&'a self) -> Self::BitIterator<'a> {
		BitsToBoolIterator::new(<ArrayBase<ViewRepr<&'x mut B>,Ix1>>::iter(&self))
	}
	#[inline(always)]
	fn get_bit_unchecked(&self, i: usize) -> bool {
		let entry = i / B::size();
		let bit_index = i % B::size();
		unsafe { self.uget(entry).get_bit_unchecked(bit_index) }
	}
	#[inline(always)]
	fn hamming_dist_same(&self, other: &Self) -> usize {
		<ArrayBase<ViewRepr<&'x mut B>,Ix1>>::iter(self)
		.zip(<ArrayBase<ViewRepr<&'x mut B>,Ix1>>::iter(other))
		.map(|(a,b)| a.hamming_dist(b))
		.sum()
	}
	#[inline(always)]
	fn dot_prod_same(&self, other: &Self) -> usize {
		<ArrayBase<ViewRepr<&'x mut B>,Ix1>>::iter(self)
		.zip(<ArrayBase<ViewRepr<&'x mut B>,Ix1>>::iter(other))
		.map(|(a,b)| a.dot_prod(b))
		.sum()
	}
	
	#[inline(always)]
	fn count_bits(&self) -> usize {
		self.iter().map(|bits| bits.count_bits()).sum()
	}
	#[inline(always)]
	fn count_bits_range_unchecked(&self, lo: usize, hi: usize) -> usize {
		let first_item = lo / B::size();
		let last_item = (hi-1) / B::size();
		let first_bit_offset = lo % B::size();
		let last_bit_offset = (hi-1) % B::size();
		unsafe {
			if first_item == last_item {
				self.uget(first_item).count_bits_range_unchecked(first_bit_offset, last_bit_offset+1)
			} else {
				let mut cnt = 0;
				if first_bit_offset == 0 {
					cnt += self.uget(first_item).count_bits();
				} else {
					cnt += self.uget(first_item).count_bits_range_unchecked(first_bit_offset, B::size());
				}
				if last_bit_offset+1 == B::size() {
					cnt += self.uget(last_item).count_bits();
				} else {
					cnt += self.uget(last_item).count_bits_range_unchecked(0, last_bit_offset+1);
				}
				cnt += (first_item+1..last_item).map(|i| self.uget(i).count_bits()).sum::<usize>();
				cnt
			}
		}
	}
}


impl<B: Bits> BitVectorMut for Vec<B> {
	#[inline(always)]
	fn set_bit_unchecked(&mut self, i: usize, v: bool) {
		let entry = i / B::size();
		let bit_index = i % B::size();
		unsafe { self.get_unchecked_mut(entry).set_bit_unchecked(bit_index, v); }
	}
}
impl<B: Bits> BitVectorMut for Array1<B> {
	#[inline(always)]
	fn set_bit_unchecked(&mut self, i: usize, v: bool) {
		let entry = i / B::size();
		let bit_index = i % B::size();
		unsafe { self.uget_mut(entry).set_bit_unchecked(bit_index, v); }
	}
}
impl<'x, B: Bits> BitVectorMut for ArrayViewMut1<'x, B> {
	#[inline(always)]
	fn set_bit_unchecked(&mut self, i: usize, v: bool) {
		let entry = i / B::size();
		let bit_index = i % B::size();
		unsafe { self.uget_mut(entry).set_bit_unchecked(bit_index, v); }
	}
}


impl<B: Bits> BitVectorOwned for Vec<B> {
	#[inline(always)]
	fn zeros(capacity: usize) -> Self {
		let n_buckets = capacity / B::size() + (if capacity % B::size() > 0 {1} else {0});
		vec![B::zeros(); n_buckets]
	}
	#[inline(always)]
	fn ones(capacity: usize) -> Self {
		let n_buckets = capacity / B::size() + (if capacity % B::size() > 0 {1} else {0});
		vec![B::ones(); n_buckets]
	}
}
impl<B: Bits> BitVectorOwned for Array1<B> {
	#[inline(always)]
	fn zeros(capacity: usize) -> Self {
		let n_buckets = capacity / B::size() + (if capacity % B::size() > 0 {1} else {0});
		Array1::from_elem(n_buckets, B::zeros())
	}
	#[inline(always)]
	fn ones(capacity: usize) -> Self {
		let n_buckets = capacity / B::size() + (if capacity % B::size() > 0 {1} else {0});
		Array1::from_elem(n_buckets, B::ones())
	}
}

impl<B: Bits> BitVectorMutOwned for Vec<B> {
	#[inline(always)]
	fn not_same(&self) -> Self {
		self.iter().map(|v| v.not()).collect()
	}
	#[inline(always)]
	fn or_same(&self, other: &Self) -> Self {
		self.iter().zip(other.iter())
		.map(|(a, b)| a.or(b))
		.collect()
	}
	#[inline(always)]
	fn and_same(&self, other: &Self) -> Self {
		self.iter().zip(other.iter())
		.map(|(a, b)| a.and(b))
		.collect()
	}
	#[inline(always)]
	fn xor_same(&self, other: &Self) -> Self {
		self.iter().zip(other.iter())
		.map(|(a, b)| a.xor(b))
		.collect()
	}
}
impl<B: Bits> BitVectorMutOwned for Array1<B> {
	#[inline(always)]
	fn not_same(&self) -> Self {
		self.iter().map(|v| v.not()).collect()
	}
	#[inline(always)]
	fn or_same(&self, other: &Self) -> Self {
		self.iter().zip(other.iter())
		.map(|(a, b)| a.or(b))
		.collect()
	}
	#[inline(always)]
	fn and_same(&self, other: &Self) -> Self {
		self.iter().zip(other.iter())
		.map(|(a, b)| a.and(b))
		.collect()
	}
	#[inline(always)]
	fn xor_same(&self, other: &Self) -> Self {
		self.iter().zip(other.iter())
		.map(|(a, b)| a.xor(b))
		.collect()
	}
}




#[cfg(test)]
macro_rules! test_bit_vector_counting {
	($bits_type: ty, $($bts: ty),*) => {
		test_bit_vector_counting!($bits_type);
		test_bit_vector_counting!($($bts),*);
	};
	($bits_type: ty) => {
		(0..100).for_each(|_| {
			let mut arr1: Array1<$bits_type> = Array1::from_shape_simple_fn(10, random) % <$bits_type>::MAX;
			let v = arr1.to_vec();
			test_bit_vector_counting!(explicit v);
			let v = arr1.view();
			test_bit_vector_counting!(explicit v);
			let v = arr1.view_mut();
			test_bit_vector_counting!(explicit v);
		});
	};
	(explicit $vec: ident) => {
		(0..100).map(|_| (random::<usize>() % ($vec.size()-2))+1)
		.for_each(|i| {
			assert!(
				$vec.count_bits() ==
				$vec.count_bits_range_unchecked(0, i) +
				$vec.count_bits_range_unchecked(i, $vec.size())
			);
			assert!(
				$vec.count_bits() ==
				$vec.count_bits_range(0, i).unwrap() +
				$vec.count_bits_range(i, $vec.size()).unwrap()
			);
			assert!(
				$vec.count_bits_range(0, i).unwrap() ==
				$vec.count_bits_range_unchecked(0, i)
			);
			assert!(
				$vec.count_bits_range(i, $vec.size()).unwrap() ==
				$vec.count_bits_range_unchecked(i, $vec.size())
			);
		});
	};
}
#[test]
fn test_bit_vec_counts() {
	use ndarray_rand::rand::random;
	test_bit_vector_counting!(u8,u16,u32,u64,u128);
}


