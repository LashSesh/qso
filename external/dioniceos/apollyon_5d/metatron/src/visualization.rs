#[derive(Debug, Clone)]
pub struct PlotSeries {
    pub title: String,
    pub data: Vec<f64>,
}

pub fn plot_history(history: &[f64], title: &str, ylabel: &str) -> PlotSeries {
    let _ = ylabel;
    PlotSeries {
        title: title.to_string(),
        data: history.to_vec(),
    }
}

pub fn plot_multiple(series: &[Vec<f64>], labels: &[&str], title: &str) -> Vec<PlotSeries> {
    series
        .iter()
        .enumerate()
        .map(|(idx, data)| PlotSeries {
            title: format!(
                "{} - {}",
                title,
                labels.get(idx).copied().unwrap_or("Series")
            ),
            data: data.clone(),
        })
        .collect()
}

pub fn plot_fieldvectors(vectors: &[Vec<f64>], title: &str) -> PlotSeries {
    let combined: Vec<f64> = vectors.iter().flatten().copied().collect();
    PlotSeries {
        title: title.to_string(),
        data: combined,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_calls() {
        let history = plot_history(&[0.0, 1.0], "Test", "Value");
        assert_eq!(history.data.len(), 2);
        let multiple = plot_multiple(&[vec![0.0, 1.0], vec![1.0, 2.0]], &["A", "B"], "Title");
        assert_eq!(multiple.len(), 2);
        let vectors = plot_fieldvectors(&[vec![1.0, 0.0]], "Vec");
        assert_eq!(vectors.data.len(), 2);
    }
}
