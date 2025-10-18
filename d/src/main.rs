mod directories;

use diabloproject::cmd::*;
use diabloproject::describe_project::event_type::EventType;

const PROJECT_NAME_DESCRIPTION: &str = "The name of the project to run.";
const PROJECT_ARGS_DESCRIPTION: &str = "Arguments to pass to the process.";

fn run_cargo() {
    todo!("Run cargo")
}

fn main() {
    let mut cmdb = CommandTreeBuilder::new();
    cmdb.command(&[]);
    cmdb.add_positional_argument(&[], "name", PROJECT_NAME_DESCRIPTION, false, false);
    cmdb.add_positional_argument(&[], "args", PROJECT_ARGS_DESCRIPTION, true, true);
    let cmd = cmdb.build();

    let repo_url = "https://github.com/diabloproject/tools";

    let content = std::fs::read_to_string("./describe.project").unwrap();
    let mut parser = diabloproject::describe_project::Parser::new(&content);
    let project = parser.parse().expect("Failed to parse describe.project");
    let variables = project
        .get_evt(EventType::Build)
        .expect("Failed to get build event");
    let use_var = *variables
        .iter()
        .find(|x| x.name == "USE")
        .expect("Failed to find USE variable");
    match use_var.value {
        "CARGO" => {}
        other => {
            eprintln!("Unknown USE ruleset: {}", other);
            std::process::exit(1);
        }
    }
    println!("{:#?}", project);
    cmd;
}
