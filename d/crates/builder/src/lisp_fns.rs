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

    let spec = Arc::new(Spec {
        name: ensure_string(&args[0])?,
        artifacts: args
            .into_iter()
            .skip(1)
            .map(|artifact: Value| match artifact {
                Value::Foreign(any) => match any.downcast::<Artifact>() {
                    Ok(art) => Ok(art.as_ref().clone()),
                    _ => err("arguments 1..inf must be artifacts"),
                },
                _ => err("arguments 1..inf must be artifacts"),
            })
            .collect::<Result<_, _>>()?,
    });
    println!("{spec:#?}");
    let c = spec.clone();
    Ok(Value::NativeClosure(Rc::new(RefCell::new(|env, args| {
        println!("{args:#?}");
        return Ok(Value::Foreign(Rc::new(c.clone().as_ref().clone())));
    }))))
}

fn ensure_string(v: &Value) -> Result<String, RuntimeError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        v => Err(RuntimeError {
            msg: format!("Expected value to be string, not {}", v.type_name()),
        }),
    }
}

pub(crate) fn artifact<E>(_: E, args: Vec<Value>) -> Result<Value, RuntimeError> {
    let art = match &args[..] {
        [v] => Artifact {
            name: ensure_string(v)?,
            r#type: ArtifactType::Executable,
        },
        [name, ty] => {
            let name = ensure_string(name)?;
            let ty = ensure_string(ty)?;
            let r#type = match ty.as_str() {
                "executable" => ArtifactType::Executable,
                "library" => ArtifactType::Library,
                "resource" => ArtifactType::Resource,
                _ => err(format!("unknown artifact type {}", ty))?
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
    std::os::unix::fs::symlink(&executable_path, &symlink_path)?;  // Explicitly use unix symlink

    Ok(())
}
