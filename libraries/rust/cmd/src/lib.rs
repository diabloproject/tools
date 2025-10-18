pub struct PositionalArgument {
    pub name: &'static str,
    pub description: &'static str,
    pub required: bool,
    pub rest: bool,
}

pub struct NamedArgument {
    pub names: &'static [&'static str],
    pub shorthands: &'static [char],
    pub description: &'static str,
}

pub struct CommandInputRequest {
    pub positional_arguments: Vec<PositionalArgument>,
    pub named_arguments: Vec<NamedArgument>,
}

// struct CommandInputs {
//     positional_arguments: Vec<(&'static str, String)>,
//     named_arguments: Vec<(&'static str, String)>,
// }

enum CommandTreeNode {
    Section {
        name: &'static str,
        subcommands: Vec<CommandTreeNode>,
    },
    Command {
        input_request: CommandInputRequest,
    },
}

impl CommandTreeNode {
    fn command_mut(&mut self) -> Option<&mut CommandInputRequest> {
        match self {
            CommandTreeNode::Command { input_request } => Some(input_request),
            _ => None,
        }
    }
}

pub struct CommandTree {
    tree: Box<CommandTreeNode>,
}

pub struct CommandTreeBuilder {
    pub(crate) tree: CommandTreeNode,
}

impl CommandTreeBuilder {
    pub fn new() -> Self {
        Self {
            tree: CommandTreeNode::Section {
                name: "/",
                subcommands: vec![],
            },
        }
    }

    pub fn command(&mut self, at: &'static [&'static str]) {
        let subcommands = match &mut self.tree {
            CommandTreeNode::Section { subcommands, .. } => subcommands,
            _ => unreachable!(),
        };
        Self::command_inner::<true>(at, 0, subcommands);
    }

    fn command_inner<'a, const CAN_MODIFY: bool>(
        at: &'static [&'static str],
        i: usize,
        search_in: &'a mut Vec<CommandTreeNode>,
    ) -> Option<&'a mut CommandTreeNode> {
        if i == at.len() {
            if CAN_MODIFY {
                search_in.push(CommandTreeNode::Command {
                    input_request: CommandInputRequest {
                        positional_arguments: vec![],
                        named_arguments: vec![],
                    },
                });
                return search_in.last_mut();
            } else {
                for node in search_in {
                    match node {
                        CommandTreeNode::Command { .. } => return Some(node),
                        _ => continue,
                    }
                }
                return None;
            }
        }
        let next = at[i];
        let section_exists = search_in
            .iter()
            .find(|node| match node {
                CommandTreeNode::Section { name, .. } => *name == next,
                _ => false,
            })
            .is_some();
        if section_exists {
            Self::command_inner::<CAN_MODIFY>(
                at,
                i + 1,
                match search_in
                    .iter_mut()
                    .find(|node| match node {
                        CommandTreeNode::Section { name, .. } => *name == next,
                        _ => false,
                    })
                    .expect("element disappeared?")
                {
                    CommandTreeNode::Section { subcommands, .. } => subcommands,
                    _ => unreachable!(),
                },
            )
        } else {
            if !CAN_MODIFY {
                return None;
            }
            search_in.push(CommandTreeNode::Section {
                name: next,
                subcommands: vec![],
            });
            Self::command_inner::<CAN_MODIFY>(
                at,
                i + 1,
                match search_in.last_mut().expect("element disappeared?") {
                    CommandTreeNode::Section { subcommands, .. } => subcommands,
                    _ => unreachable!(),
                },
            )
        }
    }

    pub fn add_positional_argument(
        &mut self,
        at: &'static [&'static str],
        name: &'static str,
        description: &'static str,
        required: bool,
        rest: bool,
    ) {
        let subcommands = match &mut self.tree {
            CommandTreeNode::Section { subcommands, .. } => subcommands,
            _ => unreachable!(),
        };
        let node = Self::command_inner::<false>(at, 0, subcommands);
        let node = node.expect("Command was not found");
        let request = node.command_mut().expect("There is a bug =)");
        request.positional_arguments.push(PositionalArgument {
            name,
            description,
            required,
            rest,
        });
    }

    pub fn build(self) -> CommandTree {
        CommandTree {
            tree: Box::new(self.tree),
        }
    }
}
