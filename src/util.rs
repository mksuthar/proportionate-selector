/// Return the index where `x` should be inserted in `a`, assuming `a` is sorted.
///
/// This implmentation is ported from python's bisect module.
///
/// Reference: https://github.com/python/cpython/blob/main/Lib/bisect.py#L68
///
/// ```rust
/// use proportionate_selector::util::*;
///
/// let arr = [1, 2, 3, 4, 10, 12, 16, 22];
///
/// assert_eq!(bisect_left(&arr, &-1), 0);
/// assert_eq!(bisect_left(&arr, &12), 5);
/// assert_eq!(bisect_left(&arr, &13), 6);
/// assert_eq!(bisect_left(&arr, &50), 8);
///
/// ```
pub fn bisect_left<T>(a: &[T], x: &T) -> usize
where
    T: std::cmp::PartialOrd,
{
    let mut left = 0;
    let mut right = a.len();

    while left < right {
        let mid = (left + right) / 2;
        if &a[mid] < x {
            left = mid + 1
        } else {
            right = mid
        }
    }
    left
}

/// Returns i32 value from f64.
///
/// ```rust
/// use proportionate_selector::util::*;
///
/// assert_eq!(convert(0.0), 0);
/// assert_eq!(convert(1.1), 1);
/// assert_eq!(convert(2.2), 2);
/// assert_eq!(convert(2.9), 3);
///
/// ```
pub fn convert(num: f64) -> i32 {
    num.round().rem_euclid(2f64.powi(32)) as u32 as i32
}
