//! StarDist's smaller-area overlap rule, using integer polygon intersections.

use super::{RAYS, StarDistRaw};
use crate::ModelError;
use clipper2_rust::constants::{MAX_COORD, MIN_COORD};
use clipper2_rust::{FillRule, Paths64, Point64, area, intersect_64};

#[derive(Debug, Clone, Copy)]
pub struct NmsOptions {
    pub probability_threshold: f32,
    pub overlap_threshold: f32,
    /// Exclude this many native-grid pixels at each edge (upstream default 2).
    pub border: usize,
}

impl Default for NmsOptions {
    fn default() -> Self {
        Self {
            probability_threshold: 0.5,
            overlap_threshold: 0.4,
            border: 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Instance {
    pub center_yx: [f32; 2],
    pub probability: f32,
    pub distances: [f32; RAYS],
}

impl Instance {
    /// Vertices in `[x, y]` order, starting on the positive x axis.
    pub fn polygon(&self) -> [[f32; 2]; RAYS] {
        std::array::from_fn(|ray| {
            let angle = std::f64::consts::TAU * ray as f64 / RAYS as f64;
            let radius = f64::from(self.distances[ray]);
            [
                (radius * angle.cos()) as f32 + self.center_yx[1],
                (radius * angle.sin()) as f32 + self.center_yx[0],
            ]
        })
    }
}

struct Candidate {
    instance: Instance,
    paths: Paths64,
    bounds: [i64; 4],
    area: f64,
}

impl Candidate {
    fn new(instance: Instance) -> Result<Self, ModelError> {
        // The NMS C++ code uses a float32 angular step and truncates vertices
        // to integers. Public polygon coordinates use the unquantized geometry.
        let step = (std::f64::consts::TAU / RAYS as f64) as f32;
        let path: Vec<Point64> = (0..RAYS)
            .map(|ray| {
                let angle = f64::from(step * ray as f32);
                let radius = f64::from(instance.distances[ray]);
                let x = (f64::from(instance.center_yx[1]) + radius * angle.cos()) as f32;
                let y = (f64::from(instance.center_yx[0]) + radius * angle.sin()) as f32;
                // The integer limits round outward as f32; exclude those endpoints.
                if ![x, y]
                    .into_iter()
                    .all(|v| v > MIN_COORD as f32 && v < MAX_COORD as f32)
                {
                    return Err(ModelError::InvalidInput {
                        model: "stardist",
                        message: "polygon vertex outside the supported clipping coordinate range"
                            .into(),
                    });
                }
                Ok(Point64::new(x as i64, y as i64))
            })
            .collect::<Result<_, _>>()?;
        let bounds = [
            path.iter().map(|p| p.x).min().unwrap(),
            path.iter().map(|p| p.y).min().unwrap(),
            path.iter().map(|p| p.x).max().unwrap(),
            path.iter().map(|p| p.y).max().unwrap(),
        ];
        let area = area(&path).abs();
        Ok(Self {
            instance,
            paths: vec![path],
            bounds,
            area,
        })
    }

    fn suppresses(&self, other: &Self, threshold: f32) -> bool {
        let [ax0, ay0, ax1, ay1] = self.bounds;
        let [bx0, by0, bx1, by1] = other.bounds;
        if ax0 > bx1 || bx0 > ax1 || ay0 > by1 || by0 > ay1 {
            return false;
        }
        let intersection: f64 = intersect_64(&self.paths, &other.paths, FillRule::NonZero)
            .iter()
            .map(|path| area(path).abs())
            .sum();
        (intersection / (self.area.min(other.area) + 1e-10)) as f32 > threshold
    }
}

pub(super) fn suppress(
    raw: &StarDistRaw,
    options: &NmsOptions,
) -> Result<Vec<Instance>, ModelError> {
    let pixels = raw.height.checked_mul(raw.width);
    if raw.height == 0
        || raw.width == 0
        || ![1, 2].contains(&raw.grid)
        || pixels != Some(raw.probability.len())
        || pixels.and_then(|n| n.checked_mul(RAYS)) != Some(raw.distances.len())
        || raw
            .probability
            .iter()
            .chain(&raw.distances)
            .any(|v| !v.is_finite())
        || ![options.probability_threshold, options.overlap_threshold]
            .iter()
            .all(|p| (0.0..=1.0).contains(p))
    {
        return Err(ModelError::InvalidInput {
            model: "stardist",
            message: "invalid prediction dimensions, values or NMS thresholds".into(),
        });
    }
    let pixels = raw.probability.len();
    let mut indices = Vec::new();
    for y in options.border..raw.height.saturating_sub(options.border) {
        for x in options.border..raw.width.saturating_sub(options.border) {
            let index = y * raw.width + x;
            if raw.probability[index] > options.probability_threshold {
                indices.push(index);
            }
        }
    }
    // Descending score, then descending grid index for deterministic equal scores.
    indices.sort_unstable_by(|&a, &b| {
        raw.probability[b]
            .total_cmp(&raw.probability[a])
            .then_with(|| b.cmp(&a))
    });
    let mut kept: Vec<Candidate> = Vec::new();
    for index in indices {
        let candidate = Candidate::new(Instance {
            center_yx: [
                ((index / raw.width) * raw.grid) as f32,
                ((index % raw.width) * raw.grid) as f32,
            ],
            probability: raw.probability[index],
            // As StarDist's predict(): avoid nonpositive distances before NMS.
            distances: std::array::from_fn(|ray| raw.distances[ray * pixels + index].max(1e-3)),
        })?;
        if !kept
            .iter()
            .any(|prior| prior.suppresses(&candidate, options.overlap_threshold))
        {
            kept.push(candidate);
        }
    }
    Ok(kept.into_iter().map(|c| c.instance).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contained_polygon_uses_smaller_area_not_iou() {
        let circle = |radius| {
            Candidate::new(Instance {
                center_yx: [30.0, 30.0],
                probability: 0.9,
                distances: [radius; RAYS],
            })
            .unwrap()
        };
        assert!(circle(20.0).suppresses(&circle(3.0), 0.5));
        assert!(!circle(20.0).suppresses(&circle(3.0), 1.0));
    }

    #[test]
    fn thresholds_edges_and_empty_candidates_are_handled() {
        let mut raw = StarDistRaw {
            height: 8,
            width: 8,
            grid: 2,
            probability: vec![0.0; 64],
            distances: vec![2.0; 64 * RAYS],
        };
        raw.probability[0] = 0.99;
        raw.probability[3 * 8 + 3] = 0.5;
        assert!(raw.instances(&NmsOptions::default()).unwrap().is_empty());
        raw.probability[3 * 8 + 3] = 0.9;
        let kept = raw.instances(&NmsOptions::default()).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].center_yx, [6.0, 6.0]);
        assert_eq!(kept[0].polygon()[0], [8.0, 6.0]);
        raw.probability[0] = f32::NAN;
        assert!(raw.instances(&NmsOptions::default()).is_err());
    }

    #[test]
    fn finite_distances_outside_clipping_range_are_rejected() {
        let mut raw = StarDistRaw {
            height: 8,
            width: 8,
            grid: 2,
            probability: vec![0.0; 64],
            distances: vec![1e20; 64 * RAYS],
        };
        raw.probability[3 * 8 + 3] = 0.9;
        assert!(raw.instances(&NmsOptions::default()).is_err());
    }
}
