// Auto-generated contract assertions from YAML — DO NOT EDIT.
// Zero cost in release builds (debug_assert!).
// Regenerate: pv codegen contracts/ -o src/generated_contracts.rs
// Include:   #[macro_use] #[allow(unused_macros)] mod generated_contracts;

// Auto-generated from contracts/gradient-v1.yaml — DO NOT EDIT
// Contract: gradient-v1

/// Preconditions for equation `gradient_clipping`.
/// Domain-specific. Call: `contract_pre_gradient_clipping!(slice_expr)`
macro_rules! contract_pre_gradient_clipping {
    () => {{}};
    ($input:expr) => {{
        let g = &$input;
        debug_assert!(g.iter().all(|v| v.is_finite()),
            "Contract gradient_clipping: precondition violated — g.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `gradient_clipping`.
/// Call before return: `contract_post_gradient_clipping!(result_expr)`
macro_rules! contract_post_gradient_clipping {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.iter().map(|v| v * v).sum::<f32>().sqrt() <= max_norm + 1e-6, "Contract gradient_clipping: postcondition violated — result.iter().map(|v| v * v).sum::<f32>().sqrt() <= max_norm + 1e-6");
    }};
}

/// Combined pre+post contract for equation `gradient_clipping`.
macro_rules! contract_gradient_clipping {
    ($input:expr, $body:expr) => {{
        contract_pre_gradient_clipping!($input);
        let _contract_result = $body;
        contract_post_gradient_clipping!(_contract_result);
        _contract_result
    }};
}

// Total: 1 preconditions, 1 postconditions from 1 contracts
