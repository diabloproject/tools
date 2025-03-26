mod lisp_fns;
mod types;

use rust_lisp::default_env;
use rust_lisp::interpreter::eval;
use rust_lisp::model::{Env, Symbol, Value};
use rust_lisp::parser::parse;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BuildError {
    #[error("Directory {build_dir} cannot be built, since it does not contain `build.dbl` file")]
    NoBuildFileError { build_dir: String },
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error("Parse error occured: ")]
    ParseError(rust_lisp::parser::ParseError),
}

pub fn build(path: &Path) -> Result<(), BuildError> {
    let build_file = path.join("build.dbl");
    if !build_file.exists() {
        return Err(BuildError::NoBuildFileError {
            build_dir: path.to_string_lossy().into(),
        });
    }
    let build_file_content = match std::fs::read_to_string(&build_file) {
        Ok(content) => content,
        Err(e) => return Err(BuildError::IOError(e)),
    };
    let ast = parse(&build_file_content);

    let env = setup_lisp_env(path);
    for expr in ast {
        let expr = match expr {
            Ok(expr) => expr,
            Err(e) => return Err(BuildError::ParseError(e)),
        };
        let res = eval(env.clone(), &expr);
        println!("{:?}", res)
    }
    Ok(())
}

fn setup_lisp_env(path: &Path) -> Rc<RefCell<Env>> {
    let env = Rc::new(RefCell::new(default_env()));
    env.borrow_mut().define(Symbol::from("spec"), Value::NativeFunc(lisp_fns::spec));
    env.borrow_mut()
        .define(Symbol::from("artifact"), Value::NativeFunc(lisp_fns::artifact));
    env.borrow_mut()
        .define(Symbol::from("emit-artifact"), Value::NativeFunc(lisp_fns::emit_artifact));
    env.borrow_mut()
        .define(Symbol::from("buildroot"), Value::String(path.to_string_lossy().into()));
    env.borrow_mut()
        .define(Symbol::from("recipe"), Value::NativeFunc(|_, args| {
            Ok(Value::NIL)
        }));
    env
}
