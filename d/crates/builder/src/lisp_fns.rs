use crate::types::*;
use rust_lisp::model::{Env, RuntimeError, Symbol, Value};
use std::cell::RefCell;
use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;

fn err<T>(msg: impl Into<String>) -> Result<T, RuntimeError> {
    Err(RuntimeError { msg: msg.into() })
}

fn check(f: impl Fn() -> bool, msg: impl Into<String>) -> Result<(), RuntimeError> {
    if !f() {
        return Err(RuntimeError { msg: msg.into() });
    }
    Ok(())
}

macro_rules! check {
    ($e: expr, $msg: literal) => {
        check(|| $e, $msg)?
    };
}

pub(crate) fn spec<E>(_: E, args: Vec<Value>) -> Result<Value, RuntimeError> {
    check!(args.len() != 0, "expected at least one argument");

    let spec = Spec {
        name: ensure_string(&args[0])?.as_str().try_into().map_err(|_| RuntimeError {
            msg: "Name must be shorter than 64 bytes".to_string(),
        })?,
        artifacts: <Vec<Artifact> as AsRef<[Artifact]>>::as_ref(
            &args
                .into_iter()
                .skip(1)
                .map(|artifact: Value| match artifact {
                    Value::Foreign(any) => match any.downcast::<Artifact>() {
                        Ok(art) => Ok(art.as_ref().clone()),
                        _ => err("arguments 1..inf must be artifacts"),
                    },
                    _ => err("arguments 1..inf must be artifacts"),
                })
                .collect::<Result<Vec<Artifact>, RuntimeError>>()?,
        )
        .try_into()
        .map_err(|_| RuntimeError {
            msg: "You can have at most 16 artifacts in one build.".to_string(),
        })?,
    };
    Ok(Value::NativeClosure(Rc::new(RefCell::new(move |env, args| {
        let mut recipes = vec![];
        for arg in args {
            match arg {
                Value::Foreign(any) if any.is::<Recipe>() => {
                    let recipe = any.downcast::<Recipe>().unwrap().as_ref().clone();
                    recipes.push(recipe);
                }
                _ => Err(rte("Argument must be a recipe."))?,
            }
        }
        return Ok(Value::Foreign(Rc::new((spec, recipes))));
    }))))
}

pub(crate) fn recipe<E>(_: E, args: Vec<Value>) -> Result<Value, RuntimeError> {
    check!(args.len() == 2, "expected exactly 2 arguments: name (string), recipe (function)");
    let name: StaticString<64> = ensure_string(&args[0])?
        .as_str()
        .try_into()
        .map_err(|_| rte("name must be shorter than 64 bytes"))?;
    let Value::Lambda(function) = args[1].clone() else {
        return Err(rte("second argument must be a lambda function"));
    };
    Ok(Value::Foreign(Rc::new(Recipe { name, function })))
}

fn ensure_string(v: &Value) -> Result<String, RuntimeError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        v => Err(rte(format!("Expected value to be string, not {}", v.type_name()))),
    }
}

pub(crate) fn artifact<E>(_: E, args: Vec<Value>) -> Result<Value, RuntimeError> {
    let art = match &args[..] {
        [v] => Artifact {
            name: ensure_string(v)?.as_str().try_into().map_err(|_| RuntimeError {
                msg: "Name must be shorter than 64 bytes".to_string(),
            })?,
            r#type: ArtifactType::Executable,
        },
        [name, ty] => {
            let name = ensure_string(name)?.as_str().try_into().map_err(|_| RuntimeError {
                msg: "Name must be shorter than 64 bytes".to_string(),
            })?;
            let ty = ensure_string(ty)?;
            let r#type = match ty.as_str() {
                "executable" => ArtifactType::Executable,
                "library" => ArtifactType::Library,
                "resource" => ArtifactType::Resource,
                _ => err(format!("unknown artifact type {}", ty))?,
            };

            Artifact { name, r#type }
        }
        [] => err("need to provide at least name")?,
        _ => err("too many arguments")?,
    };
    Ok(Value::Foreign(Rc::new(art)))
}

pub(crate) fn cargo_build(env: Rc<RefCell<Env>>, args: Vec<Value>) -> Result<Value, RuntimeError> {
    check!(args.len() == 1, "expected exactly one argument: package name");
    let package_name = ensure_string(&args[0])?;
    let buildroot = env.borrow().get(&Symbol::from("buildroot")).unwrap();

    let mut command = Command::new("cargo");
    command.arg("build");
    command.arg("--package").arg(package_name);
    command.arg("--profile").arg("release");
    command.current_dir(PathBuf::from(ensure_string(&buildroot)?));

    let output = command.output();
    check!(output.is_ok() && output.as_ref().unwrap().status.success(), "Cargo build failed");
    Ok(Value::NIL)
}

pub(crate) fn emit_artifact(env: Rc<RefCell<Env>>, args: Vec<Value>) -> Result<Value, RuntimeError> {
    check!(args.len() == 2, "expected two arguments: artifact name, ArtifactPath");
    let artifact_name = ensure_string(&args[0])?;

    let artifact_path = match &args[1] {
        Value::Foreign(any) => match any.downcast_ref::<ArtifactPath>() {
            Some(path) => path.clone(),
            None => err("second argument must be ArtifactPath")?,
        },
        _ => err("second argument must be ArtifactPath")?,
    };

    Ok(Value::NIL)
}

use crate::utils::rte;
use std::fs;
use std::sync::Arc;

fn deploy_executable(file_path: &str, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Construct paths
    let home_dir = match home::home_dir() {
        Some(path) => path,
        None => return Err("Could not find home directory".into()),
    };

    let dp_dir = home_dir.join(".dp");
    let bin_dir = dp_dir.join("bin");
    let artifacts_dir = dp_dir.join("artifacts");
    let executables_dir = artifacts_dir.join("executables");
    let libraries_dir = artifacts_dir.join("libraries");
    let resources_dir = artifacts_dir.join("resources");

    let executable_path = executables_dir.join(name);
    let symlink_path = bin_dir.join(name);

    // Create directories if they don't exist
    fs::create_dir_all(&bin_dir)?;
    fs::create_dir_all(&executables_dir)?;

    // Copy the executable
    fs::copy(file_path, &executable_path)?;

    // Create the symlink.  Important:  Need to remove the symlink if it exists.
    if symlink_path.exists() {
        fs::remove_file(&symlink_path)?;
    }
    std::os::unix::fs::symlink(&executable_path, &symlink_path)?; // Explicitly use unix symlink

    Ok(())
}
