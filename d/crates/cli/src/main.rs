use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
enum Commands {
    Build(BuildCommandArguments),
}

#[derive(Args, Debug)]
struct BuildCommandArguments {
    path: PathBuf,
}

#[derive(Parser, Debug)]
struct Arguments {
    #[clap(subcommand)]
    command: Commands,
}

fn build(args: BuildCommandArguments) {
    let result = builder::build(&args.path);
    println!("{:?}", result)
}

fn main() {
    let args = Arguments::parse();
    match args.command {
        Commands::Build(build_command) => build(build_command),
    }
}
