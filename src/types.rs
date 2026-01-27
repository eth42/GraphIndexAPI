use ndarray_linalg::{Lapack, Scalar};
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
#[test]
fn test_vfma() {
	use rand::random;
	let d = 47;
	let v1_32: Vec<f32> = (0..d).map(|_| random()).collect();
	let v2_32: Vec<f32> = (0..d).map(|_| random()).collect();
	// let v1_16: Vec<f64> = v1_32.iter().cloned().map(|v| v as f16).collect();
	// let v2_16: Vec<f64> = v2_32.iter().cloned().map(|v| v as f16).collect();
	let v1_64: Vec<f64> = v1_32.iter().cloned().map(|v| v as f64).collect();
	let v2_64: Vec<f64> = v2_32.iter().cloned().map(|v| v as f64).collect();
	let true_dist_32: f32 = v1_32.iter().zip(v2_32.iter()).map(|(&a, &b)| a-b).map(|v|v*v).sum();
	let true_dist_64: f64 = v1_64.iter().zip(v2_64.iter()).map(|(&a, &b)| a-b).map(|v|v*v).sum();
	// vec![
	// 	<f16 as VFMASqEuc<2>>::sq_euc,
	// 	<f16 as VFMASqEuc<4>>::sq_euc,
	// 	<f16 as VFMASqEuc<8>>::sq_euc,
	// 	<f16 as VFMASqEuc<16>>::sq_euc,
	// ].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
	// 	let dist = fun(v1_16.as_slice(), v2_16.as_slice(), v1_16.len()) as f32;
	// 	assert!((true_dist-dist).abs() < 1e-4, "f16x{:?}: {:?} != {:?}", lanes, true_dist, dist);
	// });
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
trait_combiner!(Float: (VFMASqEuc<2>)+(VFMASqEuc<4>)+(VFMASqEuc<8>)+(VFMASqEuc<16>)+Scalar+Lapack+Number+(num::Float));

make_num_variants!(Number, Integer, UnsignedInteger, SignedInteger, Float);

