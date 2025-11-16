//! S7 permutation space generation

/// Generate all 5040 permutations of S7
pub fn generate_s7_permutations() -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut arr = vec![0, 1, 2, 3, 4, 5, 6];
    generate_permutations(&mut arr, 0, &mut result);
    result
}

/// Helper function to generate permutations recursively
fn generate_permutations(arr: &mut [usize], start: usize, result: &mut Vec<Vec<usize>>) {
    if start == arr.len() {
        result.push(arr.to_vec());
        return;
    }

    for i in start..arr.len() {
        arr.swap(start, i);
        generate_permutations(arr, start + 1, result);
        arr.swap(start, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_count() {
        let perms = generate_s7_permutations();
        assert_eq!(perms.len(), 5040); // 7! = 5040
    }

    #[test]
    fn test_permutation_uniqueness() {
        let perms = generate_s7_permutations();
        let mut set = std::collections::HashSet::new();

        for perm in perms {
            let key = format!("{:?}", perm);
            assert!(!set.contains(&key), "Duplicate permutation found");
            set.insert(key);
        }
    }

    #[test]
    fn test_permutation_validity() {
        let perms = generate_s7_permutations();

        for perm in perms {
            assert_eq!(perm.len(), 7);

            // Check all elements 0..7 are present
            let mut sorted = perm.clone();
            sorted.sort();
            assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6]);
        }
    }
}
