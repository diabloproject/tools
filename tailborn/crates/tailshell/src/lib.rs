use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NodeId(usize);

#[derive(Debug, Clone)]
enum NodeType {
    Command {
        program: String,
        args: Vec<String>,
    },
    FsNode {
        mode: FsNodeMode,
        properties: HashMap<String, String>,
        var_name: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum FsNodeMode {
    Read,
    Write,
}

#[derive(Debug, Clone)]
pub struct Node {
    id: NodeId,
    node_type: NodeType,
    line_number: usize,
}

#[derive(Debug, Clone)]
enum EdgeType {
    Execution,
    DataFlow(String),   // The variable name flowing between nodes
    Dependency(String), // File dependency between commands
}

#[derive(Debug, Clone)]
pub struct Edge {
    from: NodeId,
    to: NodeId,
    edge_type: EdgeType,
}

#[derive(Debug)]
pub struct TailshellGraph {
    nodes: HashMap<NodeId, Node>,
    edges: Vec<Edge>,
    variables: HashMap<String, NodeId>, // Maps variable names to their defining nodes
    commands: Vec<NodeId>,              // Store commands in order of appearance
    temp_files: HashSet<String>,        // Track temp files created by commands
}

impl TailshellGraph {
    fn new() -> Self {
        TailshellGraph {
            nodes: HashMap::new(),
            edges: Vec::new(),
            variables: HashMap::new(),
            commands: Vec::new(),
            temp_files: HashSet::new(),
        }
    }

    fn add_node(&mut self, node_type: NodeType, line_number: usize) -> NodeId {
        let id = NodeId(self.nodes.len());
        let node = Node {
            id: id.clone(),
            node_type,
            line_number,
        };
        self.nodes.insert(id.clone(), node);
        id
    }

    fn add_edge(&mut self, from: &NodeId, to: &NodeId, edge_type: EdgeType) {
        let edge = Edge {
            from: from.clone(),
            to: to.clone(),
            edge_type,
        };
        self.edges.push(edge);
    }
}

pub fn parse_tailshell(content: &str) -> Result<TailshellGraph, String> {
    struct Statement(Vec<String>);
    struct Declaration {
        variable: String,
        statement: Statement;
    }
    let mut body: Vec<Statement> = vec![];
    let mut declare: Vec<Declaration> = vec![];


    enum TokenizationState {
        None,
        Word(String),
        Punctuation(String),
        Literal(String)
    }

    enum Token {
        Word(String),
        Punctiation(String),
        Literal(String)
    }

    fn tokenize(state: TokenizationState, c: char) -> TokenizationState {

    }

    #[derive(PartialEq)]
    enum StatementState {}

    #[derive(PartialEq)]
    enum BodyState {
        Root,
        Statement(StatementState)
    }

    #[derive(PartialEq)]
    enum DeclareState {
        Root,
        OnLet(String),
        OnVarName(String),
        OnEq(String),
        OnStatement(String, StatementState)
    }

    #[derive(PartialEq)]
    enum GlobalState {
        Root,
        OnDeclare(String),
        Declare(DeclareState),
        OnBody(String),
        Body(BodyState),
    }

    fn parse(state: GlobalState, c: char) -> Result<GlobalState, String> {
        Ok(match state {
            GlobalState::Root if c.is_whitespace() => state,
            GlobalState::Root if c == 'd' => GlobalState::OnDeclare("d".into()),
            GlobalState::Root if c == 'b' => GlobalState::OnBody("b".into()),
            GlobalState::Root => Err("unexpected token")?,

            GlobalState::Declare(declare_state) => todo!(),
            GlobalState::Body(body_state) => todo!(),
            GlobalState::OnDeclare(_) => todo!(),
            GlobalState::OnBody(_) => todo!(),
        })
    }
    let mut state = GlobalState::Root;
    let mut position = 0;
    for c in content.chars() {
        state = parse(state, c)?
    }
    if state != GlobalState::Root {
        return Err("early EOF".to_owned())
    }

    todo!("collect the state into the graph")
}


#[test]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let script_content = r#"
declare {
    // request from the graph executor a mov file with a write mode on, that we will write to
    let output = fsnode w ext:mov;
    // request from the graph executor a mov file with a write mode on, that we will read from
    let input = fsnode r ext:mov;
}

body {
    // ffmpeg will receive absolute paths, so we will have .mov in .aac out, aka we will extract the audio
    ffmpeg -i @input -c copy tmp.aac;
    ls 1;
    // reassemble to output
    ffmpeg -i @input -i tmp.aac -c copy @output
}
"#;

    let graph = match parse_tailshell(script_content) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error parsing tailshell script: {}", e);
            return Err(e.into());
        }
    };

    eprintln!("{:#?}", graph);
    println!("Graph visualization written to tailshell_graph.dot");
    println!("To generate the graph image, run:");
    println!("  dot -Tpng tailshell_graph.dot -o tailshell_graph.png");

    Ok(())
}
