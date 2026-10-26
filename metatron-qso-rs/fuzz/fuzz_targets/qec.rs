//! Detector circuits from bytes: building, sampling, error models and
//! export never panic, and the union-find decoder explains every syndrome
//! that some set of matching-graph edges produces.
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::qec::{
    DetectorCircuit, MatchingGraph, Noise, UnionFindDecoder, detector_error_model,
    sample_detectors,
};

fuzz_target!(|data: &[u8]| {
    let Some((&first, rest)) = data.split_first() else {
        return;
    };
    let n = 1 + usize::from(first % 6);
    let mut c = DetectorCircuit::new(n).unwrap();
    let mut records = 0usize;
    let mut chunks = rest.chunks_exact(3);
    for op in chunks.by_ref().take(200) {
        let (q, q2, x) = (usize::from(op[1]) % n, usize::from(op[2]) % n, op[2]);
        let p = f64::from(x) / 255.0 * 0.2;
        let _ = match op[0] % 10 {
            0 => c.h(q).map(|_| ()),
            1 => c.s(q).map(|_| ()),
            2 => c.cx(q, q2).map(|_| ()),
            3 => c.measure(q, p).map(|_| records += 1),
            4 => c.reset(q).map(|_| ()),
            5 => c.noise(Noise::Depolarize1(p), &[q]).map(|_| ()),
            6 => c.noise(Noise::Depolarize2(p), &[q, q2]).map(|_| ()),
            7 => c.noise(Noise::XError(p), &[q]).map(|_| ()),
            8 if records > 0 => {
                let a = records - 1 - usize::from(op[1]) % records;
                let b = records - 1 - usize::from(op[2]) % records;
                c.detector(&[a, b]).map(|_| ())
            }
            9 if records > 0 => c.observable(usize::from(op[1] % 3), &[records - 1]).map(|_| ()),
            _ => Ok(()),
        };
    }
    let _ = c.to_stim().expect("every circuit exports");
    let _ = sample_detectors(&c, 100, 1);
    let _ = detector_error_model(&c, false).expect("every circuit has a model");
    let Ok(dem) = detector_error_model(&c, true) else {
        return;
    };
    let Ok(graph) = MatchingGraph::from_dem(&dem) else {
        return;
    };
    let nd = graph.n_detectors();
    let edges = graph.edges().to_vec();
    let mut decoder = UnionFindDecoder::new(graph);
    // Syndromes of edge sets chosen by the remaining bytes.
    for pick in chunks.remainder().iter().chain(rest.iter().take(16)) {
        if edges.is_empty() {
            break;
        }
        let mut fired = vec![false; nd];
        for k in 0..=usize::from(pick % 4) {
            let e = &edges[(usize::from(*pick) * 7 + k * 13) % edges.len()];
            fired[e.a] ^= true;
            if e.b < nd {
                fired[e.b] ^= true;
            }
        }
        let fired: Vec<usize> = (0..nd).filter(|&d| fired[d]).collect();
        decoder
            .decode(&fired)
            .unwrap_or_else(|e| panic!("syndrome {fired:?} of graph edges not explained: {e}"));
    }
});
