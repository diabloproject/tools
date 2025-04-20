use rust_lisp::model::RuntimeError;

pub(crate) fn rte(v: impl Into<String>) -> RuntimeError {
    RuntimeError {
        msg: v.into(),
    }
}