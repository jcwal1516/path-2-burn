//! OpenCV-compatible `normalize`, `Sobel` and 3x3 `GaussianBlur`.
//!
//! Accumulation order and rounding follow OpenCV's separable filter engine
//! (row pass, then column pass) so the watershed energy is bit-identical.

/// `cv::borderInterpolate` for `BORDER_REFLECT_101` (`gfedcb|abcdefgh|gfedcba`).
pub(crate) fn reflect101(p: isize, len: usize) -> usize {
    let len = len as isize;
    if len <= 1 {
        return 0;
    }
    let mut p = p;
    while p < 0 || p >= len {
        p = if p < 0 { -p } else { 2 * (len - 1) - p };
    }
    p as usize
}

/// Scale and shift of `cv::normalize(NORM_MINMAX, 0, 1, CV_32F)`.
///
/// For a `CV_32F` destination OpenCV rounds the scale to f32 first and derives
/// the shift from the rounded scale.
fn minmax_scale_shift(min: f64, max: f64) -> (f64, f64) {
    let scale = if max - min > f64::EPSILON {
        1.0 / (max - min)
    } else {
        0.0
    };
    let scale = f64::from(scale as f32);
    let shift = f64::from(0.0f32 - (min * scale) as f32);
    (scale, shift)
}

fn min_max<T: Copy + PartialOrd>(values: &[T]) -> Option<(T, T)> {
    let first = *values.first()?;
    Some(values.iter().fold((first, first), |(lo, hi), &v| {
        (if v < lo { v } else { lo }, if v > hi { v } else { hi })
    }))
}

/// `cv2.normalize(src_f32, NORM_MINMAX, 0, 1, CV_32F)`; OpenCV's f32 convert
/// kernel evaluates `src * scale + shift` as a fused multiply-add in f32.
pub(crate) fn normalize_minmax_f32(src: &[f32]) -> Vec<f32> {
    let Some((lo, hi)) = min_max(src) else {
        return Vec::new();
    };
    let (scale, shift) = minmax_scale_shift(f64::from(lo), f64::from(hi));
    let (a, b) = (scale as f32, shift as f32);
    src.iter().map(|&v| v.mul_add(a, b)).collect()
}

/// `cv2.normalize(src_f64, NORM_MINMAX, 0, 1, CV_32F)`; the 64F->32F convert
/// kernel evaluates the affine map in f64 (fused) and rounds once to f32.
pub(crate) fn normalize_minmax_f64_to_f32(src: &[f64]) -> Vec<f32> {
    let Some((lo, hi)) = min_max(src) else {
        return Vec::new();
    };
    let (scale, shift) = minmax_scale_shift(lo, hi);
    src.iter()
        .map(|&v| v.mul_add(scale, shift) as f32)
        .collect()
}

/// Integer Sobel kernel of `cv2.getDerivKernels(.., ksize, normalize=False)`
/// for one axis: binomial smoothing differentiated `order` times.
pub(crate) fn sobel_kernel(ksize: usize, order: usize) -> Vec<f64> {
    let mut k = vec![0i64; ksize + 1];
    k[0] = 1;
    for _ in 0..ksize.saturating_sub(order + 1) {
        let mut old = k[0];
        for j in 1..=ksize {
            let new = k[j] + k[j - 1];
            k[j - 1] = old;
            old = new;
        }
    }
    for _ in 0..order {
        let mut old = -k[0];
        for j in 1..=ksize {
            let new = k[j - 1] - k[j];
            k[j - 1] = old;
            old = new;
        }
    }
    k.truncate(ksize);
    k.into_iter().map(|v| v as f64).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Axis {
    X,
    Y,
}

/// `cv2.Sobel(src_f32, CV_64F, dx, dy, ksize)` with `BORDER_REFLECT_101`.
pub(crate) fn sobel(
    src: &[f32],
    width: usize,
    height: usize,
    axis: Axis,
    ksize: usize,
) -> Vec<f64> {
    let deriv = sobel_kernel(ksize, 1);
    let smooth = sobel_kernel(ksize, 0);
    let (row_k, col_k, col_sym) = match axis {
        Axis::X => (deriv, smooth, Symmetry::Even),
        Axis::Y => (smooth, deriv, Symmetry::Odd),
    };
    let src: Vec<f64> = src.iter().map(|&v| f64::from(v)).collect();
    let rows = row_pass(&src, width, height, &row_k);
    column_pass(&rows, width, height, &col_k, col_sym)
}

/// `cv2.GaussianBlur(src_f64, (3, 3), 0)`: with sigma 0 and ksize 3 OpenCV
/// uses its fixed `[0.25, 0.5, 0.25]` table rather than a sampled Gaussian.
pub(crate) fn gaussian_blur3(src: &[f64], width: usize, height: usize) -> Vec<f64> {
    const K: [f64; 3] = [0.25, 0.5, 0.25];
    let rows = row_pass(src, width, height, &K);
    column_pass(&rows, width, height, &K, Symmetry::Even)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Symmetry {
    Even,
    Odd,
}

/// OpenCV `RowFilter`: plain left-to-right accumulation over the kernel taps.
fn row_pass(src: &[f64], width: usize, height: usize, kernel: &[f64]) -> Vec<f64> {
    let radius = kernel.len() / 2;
    let mut out = vec![0.0f64; width * height];
    let mut ext = vec![0.0f64; width + 2 * radius];
    for (row, dst) in src.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        for (i, e) in ext.iter_mut().enumerate() {
            *e = row[reflect101(i as isize - radius as isize, width)];
        }
        for (d, &e) in dst.iter_mut().zip(&ext) {
            *d = kernel[0] * e;
        }
        for (tap, &k) in kernel.iter().enumerate().skip(1) {
            for (d, &e) in dst.iter_mut().zip(&ext[tap..tap + width]) {
                *d += k * e;
            }
        }
    }
    out
}

/// OpenCV `SymmColumnFilter`: centre tap first, then mirrored row pairs are
/// summed (even) or differenced (odd) and accumulated. The opencv-python arm64
/// build contracts `s += k * (a + b)` into a fused multiply-add; fusing here is
/// what makes the Sobel output bit-identical to the reference.
fn column_pass(
    src: &[f64],
    width: usize,
    height: usize,
    kernel: &[f64],
    sym: Symmetry,
) -> Vec<f64> {
    let radius = kernel.len() / 2;
    let row = |y: isize| {
        let y = reflect101(y, height);
        &src[y * width..(y + 1) * width]
    };
    let mut out = vec![0.0f64; width * height];
    for (y, dst) in out.chunks_exact_mut(width).enumerate() {
        let y = y as isize;
        match sym {
            Symmetry::Even => {
                let centre = kernel[radius];
                for (d, &c) in dst.iter_mut().zip(row(y)) {
                    *d = centre * c + 0.0;
                }
            }
            Symmetry::Odd => dst.fill(0.0),
        }
        for k in 1..=radius {
            let f = kernel[radius + k];
            let (below, above) = (row(y + k as isize), row(y - k as isize));
            let pairs = dst.iter_mut().zip(below.iter().zip(above));
            match sym {
                Symmetry::Even => pairs.for_each(|(d, (&b, &a))| *d = f.mul_add(b + a, *d)),
                Symmetry::Odd => pairs.for_each(|(d, (&b, &a))| *d = f.mul_add(b - a, *d)),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect101_matches_opencv_border_interpolate() {
        let got: Vec<usize> = (-4..9).map(|p| reflect101(p, 5)).collect();
        assert_eq!(got, vec![4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0]);
        assert_eq!(reflect101(-3, 1), 0);
    }

    #[test]
    fn sobel_kernels_match_get_deriv_kernels_21() {
        // cv2.getDerivKernels(1, 0, 21, normalize=False)
        let deriv = [
            -1.0, -18.0, -152.0, -798.0, -2907.0, -7752.0, -15504.0, -23256.0, -25194.0, -16796.0,
            0.0, 16796.0, 25194.0, 23256.0, 15504.0, 7752.0, 2907.0, 798.0, 152.0, 18.0, 1.0,
        ];
        let smooth = [
            1.0, 20.0, 190.0, 1140.0, 4845.0, 15504.0, 38760.0, 77520.0, 125970.0, 167960.0,
            184756.0, 167960.0, 125970.0, 77520.0, 38760.0, 15504.0, 4845.0, 1140.0, 190.0, 20.0,
            1.0,
        ];
        assert_eq!(sobel_kernel(21, 1), deriv);
        assert_eq!(sobel_kernel(21, 0), smooth);
        // cv2.getDerivKernels(1, 0, 5, normalize=False)
        assert_eq!(sobel_kernel(5, 1), [-1.0, -2.0, 0.0, 2.0, 1.0]);
        assert_eq!(sobel_kernel(5, 0), [1.0, 4.0, 6.0, 4.0, 1.0]);
    }

    fn f32_bits(bits: &[u32]) -> Vec<f32> {
        bits.iter().map(|&b| f32::from_bits(b)).collect()
    }

    fn f64_bits(bits: &[u64]) -> Vec<f64> {
        bits.iter().map(|&b| f64::from_bits(b)).collect()
    }

    fn assert_bits_eq<T: Copy + std::fmt::Debug>(
        got: &[T],
        expected: &[T],
        bits: impl Fn(T) -> u64,
    ) {
        let got_bits: Vec<u64> = got.iter().map(|&v| bits(v)).collect();
        let exp_bits: Vec<u64> = expected.iter().map(|&v| bits(v)).collect();
        assert_eq!(got_bits, exp_bits, "got {got:?}, expected {expected:?}");
    }

    // Bit patterns below come from opencv-python 5.0.0 (arm64):
    //   src = np.arange(20, dtype=np.float32).reshape(4, 5) ** 1.5 / 7; src[1, 2] = -3.25
    //   cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=21) / (0, 1)
    const SOBEL_SRC: [u32; 20] = [
        0, 1041385765, 1053745430, 1061029853, 1066551589, 1070362821, 1074159420, 3226468352,
        1078911254, 1081531246, 1083215791, 1084671842, 1086195677, 1087784403, 1089435472,
        1090832826, 1091717413, 1092630093, 1093570026, 1094536446,
    ];

    #[test]
    fn sobel21_is_bit_identical_to_opencv() {
        let src = f32_bits(&SOBEL_SRC);
        let x = [
            0,
            4769280499611728474,
            4771262796642172521,
            4769448446539995485,
            0,
            0,
            4769351534482174007,
            4771338160979916651,
            4769499745829883187,
            0,
            0,
            4769462097709363562,
            4771488888332615883,
            4769602343279027026,
            0,
            0,
            4769517378952920979,
            4771564251347570988,
            4769653641438283163,
            0,
        ];
        let y = [
            0,
            0,
            0,
            0,
            0,
            4774064666771828034,
            4774092307763644102,
            4774157630924332237,
            4774220962738148154,
            4774246612383092004,
            4774064782823916577,
            4774092423445695284,
            4774157745574951544,
            4774221076162057164,
            4774246725241685231,
            0,
            0,
            0,
            0,
            0,
        ];
        assert_bits_eq(&sobel(&src, 5, 4, Axis::X, 21), &f64_bits(&x), f64::to_bits);
        assert_bits_eq(&sobel(&src, 5, 4, Axis::Y, 21), &f64_bits(&y), f64::to_bits);
    }

    #[test]
    fn gaussian3_is_bit_identical_to_opencv() {
        // d = (np.arange(12.0).reshape(3, 4) * 0.37) ** 2 - 1.1; cv2.GaussianBlur(d, (3, 3), 0)
        let src = f64_bits(&[
            13831004815617530266,
            13830222090002293274,
            13826522833268371149,
            4593927424376740552,
            4607589544206331700,
            4612412223867301397,
            4615803209206730009,
            4618000177694952022,
            4620312213153653352,
            4621812868844488544,
            4623277157968332719,
            4624895582789423649,
        ]);
        let expected = f64_bits(&[
            4599750578694930614,
            4604913955667710888,
            4610363986756773293,
            4612258088170054639,
            4613491173748028683,
            4615032530720496236,
            4617306567057341625,
            4618231381240822156,
            4617614838451835136,
            4618693788332562423,
            4620849520736696324,
            4621466063525683344,
        ]);
        assert_bits_eq(&gaussian_blur3(&src, 4, 3), &expected, f64::to_bits);
    }

    #[test]
    fn normalize_is_bit_identical_to_opencv() {
        // cv2.normalize(x, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        let src = [0.3f32, -1.7, 2.9, 0.1, 1.234567];
        let expected = f32_bits(&[1054776276, 827408528, 1065353216, 1053317387, 1059279014]);
        assert_bits_eq(&normalize_minmax_f32(&src), &expected, |v: f32| {
            u64::from(v.to_bits())
        });
        let src = [1e5, -2.5e4, 3.3333e4, 0.0, -7.77e4];
        // The minimum maps to a tiny negative value, as in OpenCV.
        let expected = f32_bits(&[1065353216, 1050138538, 1059058975, 1054859195, 2963406848]);
        assert_bits_eq(&normalize_minmax_f64_to_f32(&src), &expected, |v: f32| {
            u64::from(v.to_bits())
        });
        assert_eq!(normalize_minmax_f32(&[2.5; 3]), [0.0; 3]);
    }
}
