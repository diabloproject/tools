use crate::block::{Block, Variable};
use crate::event_type::EventType;
use stdd::statik::StaticVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Project<'src> {
    pub blocks: StaticVec<Block<'src>, 16>,
}

impl<'src> Project<'src> {
    pub fn get_evt(&self, ty: EventType) -> Option<&[Variable<'_>]> {
        let tt = ty;
        let block = self.blocks.iter().find(|b| match b {
            Block::Event { ty, content } => tt == *ty,
            _ => false,
        });
        match block {
            Some(Block::Event { content, .. }) => Some(content),
            _ => None,
        }
    }
}
