use crate::event_type::EventType;
use stdd::statik::StaticVec;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Block<'src> {
    AutoRule {},
    Event {
        ty: EventType,
        content: StaticVec<Variable<'src>, 64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variable<'src> {
    pub name: &'src str,
    pub value: &'src str,
}
