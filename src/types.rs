use std::iter::{Sum, Product};
use std::hash::Hash;
use std::ops::{AddAssign, SubAssign};
#[cfg(feature="hdf5")]
use hdf5::H5Type;
use num::{NumCast, FromPrimitive, ToPrimitive, Zero, One};
use paste::paste;
use num_traits::{Bounded, WrappingAdd};
use std::fmt::Debug;

#[macro_export]
macro_rules! param_struct {
	/* Matching e.g. SomeParams[Debug, Clone]<F: Float> {a: F = F::one()} */
	(
		$name:ident /* Name of the parameter struct */
		$([$($derived_type:ty),*])? /* Derived types */
		$(<$($generic_names:ident : $generic_types:path)*>)? /* Generics */
		{$($field_name:ident: $field_type:ty = $field_value:expr),*$(,)?} /* Fields */
	) => { paste::paste! {
		#[derive($($($derived_type,)*)?)]
		pub struct $name$(<$($generic_names: $generic_types),*>)? {
			$(pub $field_name: $field_type),*
		}
		impl$(<$($generic_names: $generic_types),*>)? $name$(<$($generic_names),*>)? {
			pub fn new() -> Self {
				Self {
					$($field_name: $field_value),*
				}
			}
			pub fn new_full($($field_name: Option<$field_type>,)*) -> Self {
				let mut ret = Self::new();
				$(
					if $field_name.is_some() { ret.$field_name = $field_name.unwrap(); }
				)*
				ret
			}
			$(
				pub fn [<with_ $field_name>](mut self, $field_name: $field_type) -> Self {
					self.$field_name = $field_name;
					self
				}
			)*
			$(
				pub fn [<maybe_with_ $field_name>](mut self, $field_name: Option<$field_type>) -> Self {
					if $field_name.is_some() {
						self = self.[<with_ $field_name>]($field_name.unwrap());
					}
					self
				}
			)*
		}
	}};
}
pub use param_struct;

#[macro_export]
macro_rules! trait_combiner {
	// ($combination_name: ident) => {
	// 	pub trait $combination_name {}
	// 	impl<T> $combination_name for T {}
	// };
	// ($combination_name: ident $(: $t: tt $(+ $ts: tt)*)?) => {
	// 	pub trait $combination_name $(: $t $(+ $ts)*)? {}
	// 	impl<T $(: $t $(+ $ts)*)?> $combination_name for T {}
	// };
	($combination_name: ident $([$($g: tt: $gc1: tt $(+ $gcn: tt)*),+])? $(: $t: tt $(+ $ts: tt)*)?) => {
		pub trait $combination_name$(<$($g: $gc1 $(+ $gcn)*,)+>)? $(: $t $(+ $ts)*)? {}
		impl<$($($g: $gc1 $(+ $gcn)*,)+)?T $(: $t $(+ $ts)*)?> $combination_name$(<$($g,)+>)? for T {}
	};
}
pub use trait_combiner;

use crate::bits::Bits;


pub trait VFMASqEuc<const LANES: usize>: std::ops::Sub<Output=Self>+std::ops::Mul<Output=Self>+std::ops::AddAssign+std::iter::Sum+Clone+Copy+num::Zero {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		let sd = d & !(LANES - 1);
		let mut vsum = [Self::zero(); LANES];
		for i in (0..sd).step_by(LANES) {
			let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
			for j in 0..LANES {
				unsafe {
					let x = *vv.get_unchecked(j) - *cc.get_unchecked(j);
					// emulated
					// *vsum.get_unchecked_mut(j) = x.mul_add(x, *vsum.get_unchecked(j));
					// FMA
					*vsum.get_unchecked_mut(j) += x * x;
				}
			}
		}
		let mut sum = vsum.into_iter().sum::<Self>();
		if d > sd {
			sum += (sd..d)
			.map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) })
			.map(|x| x * x)
			.sum();
		}
		sum
	}
}
impl VFMASqEuc<2> for f32 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMASqEuc<4>>::sq_euc(v1, v2, d) }
}
impl VFMASqEuc<4> for f32 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 4;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm_loadu_ps(v1.get_unchecked(i) as *const Self);
				let v2 = _mm_loadu_ps(v2.get_unchecked(i) as *const Self);
				let diff = _mm_sub_ps(v1, v2);
				vsum = _mm_fmadd_ps(diff, diff, vsum);
			}
			let sum = _mm_hadd_ps(vsum, vsum);
			let sum = _mm_hadd_ps(sum, sum);
			let mut sum = _mm_cvtss_f32(sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) - *v2.get_unchecked(i))
				.map(|x| x * x)
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMASqEuc<8> for f32 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 8;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm256_loadu_ps(v1.get_unchecked(i) as *const Self);
				let v2 = _mm256_loadu_ps(v2.get_unchecked(i) as *const Self);
				let diff = _mm256_sub_ps(v1, v2);
				vsum = _mm256_fmadd_ps(diff, diff, vsum);
			}
			let sum = _mm256_hadd_ps(vsum, vsum);
			let sum = _mm256_hadd_ps(sum, sum);
			let sum_low = _mm256_castps256_ps128(sum);
			let sum_high = _mm256_extractf128_ps(sum, 1);
			let final_sum = _mm_add_ps(sum_low, sum_high);
			let mut sum = _mm_cvtss_f32(final_sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) - *v2.get_unchecked(i))
				.map(|x| x * x)
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMASqEuc<16> for f32 {
	#[cfg(not(feature="nightly-features"))]
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMASqEuc<8>>::sq_euc(v1, v2, d) }
	#[cfg(feature="nightly-features")]
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 16;
		unsafe {
			use std::arch::x86_64::*;
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let v1 = _mm256_loadu_ps(v1.get_unchecked(i) as *const Self);
				let v2 = _mm256_loadu_ps(v2.get_unchecked(i) as *const Self);
				let diff = _mm256_sub_ps(v1, v2);
				vsum = _mm256_fmadd_ps(diff, diff, vsum);
			}
			let sum = _mm256_hadd_ps(vsum, vsum);
			let sum = _mm256_hadd_ps(sum, sum);
			let sum_low = _mm256_castps256_ps128(sum);
			let sum_high = _mm256_extractf128_ps(sum, 1);
			let final_sum = _mm_add_ps(sum_low, sum_high);
			let mut sum = _mm_cvtss_f32(final_sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) - *v2.get_unchecked(i))
				.map(|x| x * x)
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMASqEuc<2> for f64 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 2;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm_setzero_pd();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm_loadu_pd(v1.get_unchecked(i) as *const Self);
				let v2 = _mm_loadu_pd(v2.get_unchecked(i) as *const Self);
				let diff = _mm_sub_pd(v1, v2);
				vsum = _mm_fmadd_pd(diff, diff, vsum);
			}
			let sum = _mm_hadd_pd(vsum, vsum);
			let mut sum = _mm_cvtsd_f64(sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) - *v2.get_unchecked(i))
				.map(|x| x * x)
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMASqEuc<4> for f64 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 4;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_pd();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm256_loadu_pd(v1.get_unchecked(i) as *const Self);
				let v2 = _mm256_loadu_pd(v2.get_unchecked(i) as *const Self);
				let diff = _mm256_sub_pd(v1, v2);
				vsum = _mm256_fmadd_pd(diff, diff, vsum);
			}
			let sum = _mm256_hadd_pd(vsum, vsum);
			let sum_low = _mm256_castpd256_pd128(sum);
			let sum_high = _mm256_extractf128_pd(sum, 1);
			let final_sum = _mm_add_pd(sum_low, sum_high);
			let mut sum = _mm_cvtsd_f64(final_sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) - *v2.get_unchecked(i))
				.map(|x| x * x)
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMASqEuc<8> for f64 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMASqEuc<4>>::sq_euc(v1, v2, d) }
}
impl VFMASqEuc<16> for f64 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMASqEuc<4>>::sq_euc(v1, v2, d) }
}
impl VFMASqEuc<2> for half::f16 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMASqEuc<8>>::sq_euc(v1, v2, d) }
}
impl VFMASqEuc<4> for half::f16 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMASqEuc<8>>::sq_euc(v1, v2, d) }
}
impl VFMASqEuc<8> for half::f16 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d);
		const LANES: usize = 8; // 8 f16 -> 8 f32 in __m256
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.as_ptr() as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.as_ptr() as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let next_i = i + LANES;
				if next_i < d {
					_mm_prefetch(v1.as_ptr().add(next_i) as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.as_ptr().add(next_i) as *const i8, _MM_HINT_T0);
				}
				let a = _mm_loadu_si128(v1.as_ptr().add(i) as *const __m128i);
				let b = _mm_loadu_si128(v2.as_ptr().add(i) as *const __m128i);
				let a = _mm256_cvtph_ps(a);
				let b = _mm256_cvtph_ps(b);
				let diff = _mm256_sub_ps(a, b);
				vsum = _mm256_fmadd_ps(diff, diff, vsum);
			}
			let lo = _mm256_castps256_ps128(vsum);
			let hi = _mm256_extractf128_ps(vsum, 1);
			let sum128 = _mm_add_ps(lo, hi);
			let sum128 = _mm_hadd_ps(sum128, sum128);
			let sum128 = _mm_hadd_ps(sum128, sum128);
			let mut sum = _mm_cvtss_f32(sum128);
			if d > sd {
				for i in sd..d {
					let x = v1.get_unchecked(i).to_f32().unwrap_unchecked() - v2.get_unchecked(i).to_f32().unwrap_unchecked();
					sum += x * x;
				}
			}
			half::f16::from_f32(sum)
		}
	}
}
impl VFMASqEuc<16> for half::f16 {
	#[inline(always)]
	fn sq_euc(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMASqEuc<8>>::sq_euc(v1, v2, d) }
}
#[test]
fn test_vfma() {
	use rand::random;
	let d = 47;
	let v1_16: Vec<half::f16> = (0..d).map(|_| half::f16::from_f32(random())).collect();
	let v2_16: Vec<half::f16> = (0..d).map(|_| half::f16::from_f32(random())).collect();
	let v1_32: Vec<f32> = (0..d).map(|_| random()).collect();
	let v2_32: Vec<f32> = (0..d).map(|_| random()).collect();
	// let v1_16: Vec<f64> = v1_32.iter().cloned().map(|v| v as f16).collect();
	// let v2_16: Vec<f64> = v2_32.iter().cloned().map(|v| v as f16).collect();
	let v1_64: Vec<f64> = v1_32.iter().cloned().map(|v| v as f64).collect();
	let v2_64: Vec<f64> = v2_32.iter().cloned().map(|v| v as f64).collect();
	let true_dist_16: half::f16 = v1_16.iter().zip(v2_16.iter()).map(|(&a, &b)| a-b).map(|v|v*v).sum();
	let true_dist_32: f32 = v1_32.iter().zip(v2_32.iter()).map(|(&a, &b)| a-b).map(|v|v*v).sum();
	let true_dist_64: f64 = v1_64.iter().zip(v2_64.iter()).map(|(&a, &b)| a-b).map(|v|v*v).sum();
	[
		<half::f16 as VFMASqEuc<2>>::sq_euc,
		<half::f16 as VFMASqEuc<4>>::sq_euc,
		<half::f16 as VFMASqEuc<8>>::sq_euc,
		<half::f16 as VFMASqEuc<16>>::sq_euc,
	].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_16.as_slice(), v2_16.as_slice(), v1_16.len()) as half::f16;
		assert!((true_dist_16-dist).to_f32().abs() < 1e-3 * true_dist_16.to_f32(), "f16x{:?}: {:?} != {:?}", lanes, true_dist_16, dist);
	});
	[
		<f32 as VFMASqEuc<2>>::sq_euc,
		<f32 as VFMASqEuc<4>>::sq_euc,
		<f32 as VFMASqEuc<8>>::sq_euc,
		<f32 as VFMASqEuc<16>>::sq_euc,
	].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_32.as_slice(), v2_32.as_slice(), v1_32.len());
		assert!((true_dist_32-dist).abs() < 1e-5, "f32x{:?}: {:?} != {:?}", lanes, true_dist_32, dist);
	});
	[
		<f64 as VFMASqEuc<2>>::sq_euc,
		<f64 as VFMASqEuc<4>>::sq_euc,
		<f64 as VFMASqEuc<8>>::sq_euc,
		<f64 as VFMASqEuc<16>>::sq_euc,
		].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_64.as_slice(), v2_64.as_slice(), v1_64.len());
		assert!((true_dist_64-dist).abs() < 1e-10, "f64x{:?}: {:?} != {:?}", lanes, true_dist_64, dist);
	});
}

pub trait VFMADotProd<const LANES: usize>: std::ops::Sub<Output=Self>+std::ops::Mul<Output=Self>+std::ops::AddAssign+std::iter::Sum+Clone+Copy+num::Zero {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		let sd = d & !(LANES - 1);
		let mut vsum = [Self::zero(); LANES];
		for i in (0..sd).step_by(LANES) {
			let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
			for j in 0..LANES {
				unsafe {
					let x = *vv.get_unchecked(j) * *cc.get_unchecked(j);
					// emulated
					// *vsum.get_unchecked_mut(j) = x.mul_add(x, *vsum.get_unchecked(j));
					// FMA
					*vsum.get_unchecked_mut(j) += x;
				}
			}
		}
		let mut sum = vsum.into_iter().sum::<Self>();
		if d > sd {
			sum += (sd..d)
			.map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) })
			.sum();
		}
		sum
	}
}
impl VFMADotProd<2> for f32 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<4>>::dot_prod(v1, v2, d) }
}
impl VFMADotProd<4> for f32 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 4;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm_loadu_ps(v1.get_unchecked(i) as *const Self);
				let v2 = _mm_loadu_ps(v2.get_unchecked(i) as *const Self);
				vsum = _mm_fmadd_ps(v1, v2, vsum);
			}
			let sum = _mm_hadd_ps(vsum, vsum);
			let sum = _mm_hadd_ps(sum, sum);
			let mut sum = _mm_cvtss_f32(sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) * *v2.get_unchecked(i))
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMADotProd<8> for f32 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 8;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm256_loadu_ps(v1.get_unchecked(i) as *const Self);
				let v2 = _mm256_loadu_ps(v2.get_unchecked(i) as *const Self);
				vsum = _mm256_fmadd_ps(v1, v2, vsum);
			}
			let sum = _mm256_hadd_ps(vsum, vsum);
			let sum = _mm256_hadd_ps(sum, sum);
			let sum_low = _mm256_castps256_ps128(sum);
			let sum_high = _mm256_extractf128_ps(sum, 1);
			let final_sum = _mm_add_ps(sum_low, sum_high);
			let mut sum = _mm_cvtss_f32(final_sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) * *v2.get_unchecked(i))
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMADotProd<16> for f32 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<8>>::dot_prod(v1, v2, d) }
}
impl VFMADotProd<2> for f64 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 2;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm_setzero_pd();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm_loadu_pd(v1.get_unchecked(i) as *const Self);
				let v2 = _mm_loadu_pd(v2.get_unchecked(i) as *const Self);
				vsum = _mm_fmadd_pd(v1, v2, vsum);
			}
			let sum = _mm_hadd_pd(vsum, vsum);
			let mut sum = _mm_cvtsd_f64(sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) * *v2.get_unchecked(i))
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMADotProd<4> for f64 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 4;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_pd();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm256_loadu_pd(v1.get_unchecked(i) as *const Self);
				let v2 = _mm256_loadu_pd(v2.get_unchecked(i) as *const Self);
				vsum = _mm256_fmadd_pd(v1, v2, vsum);
			}
			let sum = _mm256_hadd_pd(vsum, vsum);
			let sum_low = _mm256_castpd256_pd128(sum);
			let sum_high = _mm256_extractf128_pd(sum, 1);
			let final_sum = _mm_add_pd(sum_low, sum_high);
			let mut sum = _mm_cvtsd_f64(final_sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) * *v2.get_unchecked(i))
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMADotProd<8> for f64 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<4>>::dot_prod(v1, v2, d) }
}
impl VFMADotProd<16> for f64 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<4>>::dot_prod(v1, v2, d) }
}
impl VFMADotProd<2> for half::f16 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<8>>::dot_prod(v1, v2, d) }
}
impl VFMADotProd<4> for half::f16 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<8>>::dot_prod(v1, v2, d) }
}
impl VFMADotProd<8> for half::f16 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d);
		const LANES: usize = 8; // 8 f16 -> 8 f32 in __m256
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.as_ptr() as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.as_ptr() as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let next_i = i + LANES;
				if next_i < d {
					_mm_prefetch(v1.as_ptr().add(next_i) as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.as_ptr().add(next_i) as *const i8, _MM_HINT_T0);
				}
				let a = _mm_loadu_si128(v1.as_ptr().add(i) as *const __m128i);
				let b = _mm_loadu_si128(v2.as_ptr().add(i) as *const __m128i);
				let a = _mm256_cvtph_ps(a);
				let b = _mm256_cvtph_ps(b);
				vsum = _mm256_fmadd_ps(a, b, vsum);
			}
			let lo = _mm256_castps256_ps128(vsum);
			let hi = _mm256_extractf128_ps(vsum, 1);
			let sum128 = _mm_add_ps(lo, hi);
			let sum128 = _mm_hadd_ps(sum128, sum128);
			let sum128 = _mm_hadd_ps(sum128, sum128);
			let mut sum = _mm_cvtss_f32(sum128);
			if d > sd {
				for i in sd..d {
					let x = v1.get_unchecked(i).to_f32().unwrap_unchecked() * v2.get_unchecked(i).to_f32().unwrap_unchecked();
					sum += x;
				}
			}
			half::f16::from_f32(sum)
		}
	}
}
impl VFMADotProd<16> for half::f16 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<8>>::dot_prod(v1, v2, d) }
}
#[test]
fn test_vfma_dot() {
	use rand::random;
	let d = 47;
	let v1_16: Vec<half::f16> = (0..d).map(|_| half::f16::from_f32(random())).collect();
	let v2_16: Vec<half::f16> = (0..d).map(|_| half::f16::from_f32(random())).collect();
	let v1_32: Vec<f32> = (0..d).map(|_| random()).collect();
	let v2_32: Vec<f32> = (0..d).map(|_| random()).collect();
	// let v1_16: Vec<f64> = v1_32.iter().cloned().map(|v| v as f16).collect();
	// let v2_16: Vec<f64> = v2_32.iter().cloned().map(|v| v as f16).collect();
	let v1_64: Vec<f64> = v1_32.iter().cloned().map(|v| v as f64).collect();
	let v2_64: Vec<f64> = v2_32.iter().cloned().map(|v| v as f64).collect();
	let true_dist_16: half::f16 = v1_16.iter().zip(v2_16.iter()).map(|(&a, &b)| a*b).sum();
	let true_dist_32: f32 = v1_32.iter().zip(v2_32.iter()).map(|(&a, &b)| a*b).sum();
	let true_dist_64: f64 = v1_64.iter().zip(v2_64.iter()).map(|(&a, &b)| a*b).sum();
	[
		<half::f16 as VFMADotProd<2>>::dot_prod,
		<half::f16 as VFMADotProd<4>>::dot_prod,
		<half::f16 as VFMADotProd<8>>::dot_prod,
		<half::f16 as VFMADotProd<16>>::dot_prod,
	].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_16.as_slice(), v2_16.as_slice(), v1_16.len()) as half::f16;
		assert!((true_dist_16-dist).to_f32().abs() < 1e-3 * true_dist_16.to_f32(), "f16x{:?}: {:?} != {:?}", lanes, true_dist_16, dist);
	});
	[
		<f32 as VFMADotProd<2>>::dot_prod,
		<f32 as VFMADotProd<4>>::dot_prod,
		<f32 as VFMADotProd<8>>::dot_prod,
		<f32 as VFMADotProd<16>>::dot_prod,
	].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_32.as_slice(), v2_32.as_slice(), v1_32.len());
		assert!((true_dist_32-dist).abs() < 1e-5, "f32x{:?}: {:?} != {:?}", lanes, true_dist_32, dist);
	});
	[
		<f64 as VFMADotProd<2>>::dot_prod,
		<f64 as VFMADotProd<4>>::dot_prod,
		<f64 as VFMADotProd<8>>::dot_prod,
		<f64 as VFMADotProd<16>>::dot_prod,
		].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_64.as_slice(), v2_64.as_slice(), v1_64.len());
		assert!((true_dist_64-dist).abs() < 1e-10, "f64x{:?}: {:?} != {:?}", lanes, true_dist_64, dist);
	});
}


macro_rules! to_uint {
	($val: expr) => {
		std::mem::transmute::<&Self,&Self::UINTBase>($val)
	}
}
pub trait VFMAHamming<const LANES: usize>: std::ops::Sub<Output=Self>+std::ops::Mul<Output=Self>+std::ops::AddAssign+std::iter::Sum+Clone+Copy+num::Zero+NumCast {
	type UINTBase: Bits;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize {
		debug_assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		let sd = d & !(LANES - 1);
		let mut vsum = [0usize; LANES];
		for i in (0..sd).step_by(LANES) {
			let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
			for j in 0..LANES {
				unsafe {
					let x = to_uint!(vv.get_unchecked(j)).xor(to_uint!(cc.get_unchecked(j)));
					// FMA
					*vsum.get_unchecked_mut(j) += x.count_bits();
				}
			}
		}
		let mut sum = vsum.into_iter().sum::<usize>();
		if d > sd {
			sum += (sd..d)
			.map(|i| unsafe { to_uint!(v1.get_unchecked(i)).xor(to_uint!(v2.get_unchecked(i))).count_bits() })
			.sum::<usize>();
		}
		sum
	}
}
impl VFMAHamming<2> for f32 {
	type UINTBase=u32;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize { <Self as VFMAHamming<4>>::hamm_dist(v1, v2, d) }
}
impl VFMAHamming<4> for f32 {
	type UINTBase=u32;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 4;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut sum = 0u64;
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm_loadu_ps(v1.get_unchecked(i) as *const Self);
				let v2 = _mm_loadu_ps(v2.get_unchecked(i) as *const Self);
				let diff = _mm_xor_ps(v1,v2);
				let diff = _mm_castps_si128(diff);
				sum += _popcnt64(_mm_cvtsi128_si64(diff)) as u64;
				sum += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(diff, 8))) as u64;
			}
			if d > sd {
				sum += (sd..d)
				.map(|i| to_uint!(v1.get_unchecked(i)).xor(to_uint!(v2.get_unchecked(i))).count_bits() as u64)
				.sum::<u64>();
			}
			sum as usize
		}
	}
}
impl VFMAHamming<8> for f32 {
	type UINTBase=u32;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 8;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut sum = 0u64;
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm256_loadu_ps(v1.get_unchecked(i) as *const Self);
				let v2 = _mm256_loadu_ps(v2.get_unchecked(i) as *const Self);
				let diff = _mm256_xor_ps(v1,v2);
				let diff = _mm256_castps_si256(diff);
				// Extract low 128 bits (first 4 f32 → 2 u64)
				let low = _mm256_extractf128_si256(diff, 0);
				sum += _popcnt64(_mm_cvtsi128_si64(low)) as u64;
				sum += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(low, 8))) as u64;
				// Extract high 128 bits (next 4 f32 → 2 u64)  
				let high = _mm256_extractf128_si256(diff, 1);
				sum += _popcnt64(_mm_cvtsi128_si64(high)) as u64;
				sum += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(high, 8))) as u64;
			}
			if d > sd {
				sum += (sd..d)
				.map(|i| to_uint!(v1.get_unchecked(i)).xor(to_uint!(v2.get_unchecked(i))).count_bits() as u64)
				.sum::<u64>();
			}
			sum as usize
		}
	}
}
impl VFMAHamming<16> for f32 {
	type UINTBase=u32;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize { <Self as VFMAHamming<8>>::hamm_dist(v1, v2, d) }
}
impl VFMAHamming<2> for f64 {
	type UINTBase=u64;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 2;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut sum = 0u64;
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm_loadu_pd(v1.get_unchecked(i) as *const Self);
				let v2 = _mm_loadu_pd(v2.get_unchecked(i) as *const Self);
				let diff = _mm_xor_pd(v1,v2);
				let diff = _mm_castpd_si128(diff);
				sum += _popcnt64(_mm_cvtsi128_si64(diff)) as u64;
				sum += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(diff, 8))) as u64;
			}
			if d > sd {
				sum += (sd..d)
				.map(|i| to_uint!(v1.get_unchecked(i)).xor(to_uint!(v2.get_unchecked(i))).count_bits() as u64)
				.sum::<u64>();
			}
			sum as usize
		}
	}
}
impl VFMAHamming<4> for f64 {
	type UINTBase=u64;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 4;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut sum = 0u64;
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm256_loadu_pd(v1.get_unchecked(i) as *const Self);
				let v2 = _mm256_loadu_pd(v2.get_unchecked(i) as *const Self);
				let diff = _mm256_xor_pd(v1,v2);
				let diff = _mm256_castpd_si256(diff);
				// Extract low 128 bits (first 4 f32 → 2 u64)
				let low = _mm256_extractf128_si256(diff, 0);
				sum += _popcnt64(_mm_cvtsi128_si64(low)) as u64;
				sum += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(low, 8))) as u64;
				// Extract high 128 bits (next 4 f32 → 2 u64)  
				let high = _mm256_extractf128_si256(diff, 1);
				sum += _popcnt64(_mm_cvtsi128_si64(high)) as u64;
				sum += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(high, 8))) as u64;
			}
			if d > sd {
				sum += (sd..d)
				.map(|i| to_uint!(v1.get_unchecked(i)).xor(to_uint!(v2.get_unchecked(i))).count_bits() as u64)
				.sum::<u64>();
			}
			sum as usize
		}
	}
}
impl VFMAHamming<8> for f64 {
	type UINTBase=u64;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize { <Self as VFMAHamming<4>>::hamm_dist(v1, v2, d) }
}
impl VFMAHamming<16> for f64 {
	type UINTBase=u64;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize { <Self as VFMAHamming<4>>::hamm_dist(v1, v2, d) }
}
impl VFMAHamming<2> for half::f16 {
	type UINTBase=u16;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize { <Self as VFMAHamming<8>>::hamm_dist(v1, v2, d) }
}
impl VFMAHamming<4> for half::f16 {
	type UINTBase=u16;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize { <Self as VFMAHamming<8>>::hamm_dist(v1, v2, d) }
}
impl VFMAHamming<8> for half::f16 {
	type UINTBase=u16;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize {
    debug_assert!(v1.len() == d && v2.len() == d);
		const LANES: usize = 8;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut sum = 0u64;
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm_loadu_si128(v1.get_unchecked(i) as *const Self as *const __m128i);
				let v2 = _mm_loadu_si128(v2.get_unchecked(i) as *const Self as *const __m128i);
				let diff = _mm_xor_si128(v1,v2);
				sum += _popcnt64(_mm_cvtsi128_si64(diff)) as u64;
				sum += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(diff, 8))) as u64;
			}
			if d > sd {
				sum += (sd..d)
				.map(|i| to_uint!(v1.get_unchecked(i)).xor(to_uint!(v2.get_unchecked(i))).count_bits() as u64)
				.sum::<u64>();
			}
			sum as usize
		}
	}
}
impl VFMAHamming<16> for half::f16 {
	type UINTBase=u16;
	#[inline(always)]
	fn hamm_dist(v1: &[Self], v2: &[Self], d: usize) -> usize {
    debug_assert!(v1.len() == d && v2.len() == d);
		const LANES: usize = 16;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut sum = 0u64;
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm256_loadu_si256(v1.get_unchecked(i) as *const Self as *const __m256i);
				let v2 = _mm256_loadu_si256(v2.get_unchecked(i) as *const Self as *const __m256i);
				let diff = _mm256_xor_si256(v1,v2);
				// Extract low 128 bits (first 8 f16 → 2 u64)
				let low = _mm256_extractf128_si256(diff, 0);
				sum += _popcnt64(_mm_cvtsi128_si64(low)) as u64;
				sum += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(low, 8))) as u64;
				// Extract high 128 bits (next 8 f16 → 2 u64)  
				let high = _mm256_extractf128_si256(diff, 1);
				sum += _popcnt64(_mm_cvtsi128_si64(high)) as u64;
				sum += _popcnt64(_mm_cvtsi128_si64(_mm_srli_si128(high, 8))) as u64;
			}
			if d > sd {
				sum += (sd..d)
				.map(|i| to_uint!(v1.get_unchecked(i)).xor(to_uint!(v2.get_unchecked(i))).count_bits() as u64)
				.sum::<u64>();
			}
			sum as usize
		}
	}
}

#[test]
fn test_vfma_hamm() {
	use rand::random;
	use crate::bit_vectors::BitVector;
	let d = 47;
	let v1_16: Vec<half::f16> = (0..d).map(|_| half::f16::from_f32(random())).collect();
	let v2_16: Vec<half::f16> = (0..d).map(|_| half::f16::from_f32(random())).collect();
	let v1_32: Vec<f32> = (0..d).map(|_| random()).collect();
	let v2_32: Vec<f32> = (0..d).map(|_| random()).collect();
	let v1_64: Vec<f64> = v1_32.iter().cloned().map(|v| v as f64).collect();
	let v2_64: Vec<f64> = v2_32.iter().cloned().map(|v| v as f64).collect();
	let true_dist_16 = v1_16.hamming_dist(&v2_16);
	let true_dist_32 = v1_32.hamming_dist(&v2_32);
	let true_dist_64 = v1_64.hamming_dist(&v2_64);
	[
		<half::f16 as VFMAHamming<2>>::hamm_dist,
		<half::f16 as VFMAHamming<4>>::hamm_dist,
		<half::f16 as VFMAHamming<8>>::hamm_dist,
		<half::f16 as VFMAHamming<16>>::hamm_dist,
	].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_16.as_slice(), v2_16.as_slice(), v1_16.len());
		assert!(true_dist_16 == dist, "f16x{:?}: {:?} != {:?}", lanes, true_dist_16, dist);
	});
	[
		<f32 as VFMAHamming<2>>::hamm_dist,
		<f32 as VFMAHamming<4>>::hamm_dist,
		<f32 as VFMAHamming<8>>::hamm_dist,
		<f32 as VFMAHamming<16>>::hamm_dist,
	].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_32.as_slice(), v2_32.as_slice(), v1_32.len());
		assert!(true_dist_32 == dist, "f32x{:?}: {:?} != {:?}", lanes, true_dist_32, dist);
	});
	[
		<f64 as VFMAHamming<2>>::hamm_dist,
		<f64 as VFMAHamming<4>>::hamm_dist,
		<f64 as VFMAHamming<8>>::hamm_dist,
		<f64 as VFMAHamming<16>>::hamm_dist,
		].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_64.as_slice(), v2_64.as_slice(), v1_64.len());
		assert!(true_dist_64 == dist, "f64x{:?}: {:?} != {:?}", lanes, true_dist_64, dist);
	});
}

#[cfg(feature="python")]
pub mod python {
	pub trait NumpyEquivalent: numpy::Element {
		fn numpy_name() -> &'static str;
	}
	macro_rules! make_numpy_equivalent {
		($(($rust_types: ty, $numpy_names: literal)),*) => {
			$(make_numpy_equivalent!($rust_types, $numpy_names);)*
		};
		($rust_type: ty, $numpy_name: literal) => {
			impl NumpyEquivalent for $rust_type {
				fn numpy_name() -> &'static str {
					$numpy_name
				}
			}
		};
	}
	use half::f16;
	make_numpy_equivalent!(
		(f16, "float16"), (f32, "float32"), (f64, "float64"), /*(f128, "float128"), // Not yet supported it appears */
		(bool, "bool_"),
		(u8, "uint8"), (u16, "uint16"),	(u32, "uint32"), (u64, "uint64"),
		(i8, "int8"), (i16, "int16"),	(i32, "int32"), (i64, "int64")
	);
	#[cfg(target_pointer_width = "16")]
	make_numpy_equivalent!((usize, "uint16"), (isize, "int16"));
	#[cfg(target_pointer_width = "32")]
	make_numpy_equivalent!((usize, "uint32"), (isize, "int32"));
	#[cfg(target_pointer_width = "64")]
	make_numpy_equivalent!((usize, "uint64"), (isize, "int64"));
}

trait_combiner!(Static: 'static);
trait_combiner!(Sync: (std::marker::Send)+(std::marker::Sync));
#[cfg(all(feature="python", feature="hdf5"))]
trait_combiner!(Number: (python::NumpyEquivalent)+Bounded+H5Type+NumCast+FromPrimitive+ToPrimitive+Zero+One+Sum+Product+SubAssign+AddAssign+Copy+Clone+Debug+'static);
#[cfg(all(feature="python", not(feature="hdf5")))]
trait_combiner!(Number: (python::NumpyEquivalent)+Bounded+NumCast+FromPrimitive+ToPrimitive+Zero+One+Sum+Product+SubAssign+AddAssign+Copy+Clone+Debug+'static);
#[cfg(all(not(feature="python"), feature="hdf5"))]
trait_combiner!(Number: Bounded+H5Type+NumCast+FromPrimitive+ToPrimitive+Zero+One+Sum+Product+SubAssign+AddAssign+Copy+Clone+Debug+'static);
#[cfg(all(not(feature="python"), not(feature="hdf5")))]
trait_combiner!(Number: Bounded+NumCast+FromPrimitive+ToPrimitive+Zero+One+Sum+Product+SubAssign+AddAssign+Copy+Clone+Debug+'static);

macro_rules! make_num_variants {
	($($baseType:ident),*) => {
		paste! {
			$(
				trait_combiner!([<Static $baseType>]: $baseType+Static);
				trait_combiner!([<Sync $baseType>]: $baseType+Sync);
				trait_combiner!([<StaticSync $baseType>]: [<Sync $baseType>]+Static);
			)*
		}
	};
}

trait_combiner!(Integer: Number+(num::Integer));
trait_combiner!(UnsignedInteger: Hash+Integer+(num::Unsigned)+WrappingAdd);
trait_combiner!(SignedInteger: Integer+(num::Signed));
trait_combiner!(Float: (VFMASqEuc<2>)+(VFMASqEuc<4>)+(VFMASqEuc<8>)+(VFMASqEuc<16>)+(VFMADotProd<2>)+(VFMADotProd<4>)+(VFMADotProd<8>)+(VFMADotProd<16>)+(VFMAHamming<2>)+(VFMAHamming<4>)+(VFMAHamming<8>)+(VFMAHamming<16>)+Number+(num::Float));

make_num_variants!(Number, Integer, UnsignedInteger, SignedInteger, Float);

